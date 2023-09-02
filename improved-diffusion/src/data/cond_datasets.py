# from PIL import Image
# import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, PreTrainedTokenizerFast, \
    PreTrainedTokenizer
from datasets import load_dataset
import sys, os
import torch
from glob import glob
from tqdm import tqdm
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
import scipy.io as sio
import h5py


def load_data_text(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, data_args=None, 
        task_mode='roc', model=None, padding_mode='block', split='train', load_vocab=None, subtask=None
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    print('hello loading text data. ')
    print(f"task mode is? {task_mode}")

    if data_args.experiment.startswith('random') and model is None:
        model = None
    elif data_args.experiment.startswith('random') and model is not None:
        print('loading initialized random embeddings. ')

    if task_mode == 'roc' or task_mode == 'roc-aug' :
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)
    elif task_mode == 'simple-wiki':
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)

    elif task_mode == 'e2e-tgt':
        print('hello loading e2e-tgt. ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)
        
    ########################################################################################################
    #                 						    START ADD      
    ########################################################################################################
    elif task_mode == "zuco":
        print("Hello, loading Zuco")
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)
    ########################################################################################################
    #                 						    END ADD      
    ########################################################################################################
        
    elif task_mode == 'yelp':
        print('hello loading yelp ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)

    elif task_mode == 'commonGen' or task_mode == 'commonGen-aug':
        print('hello loading common-gen ')
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)

    elif task_mode == 'e2e':
        training_data, model = get_corpus_rocstory(data_args, model, image_size,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab, subtask=subtask)

    elif task_mode == 'book':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        training_data, model = get_corpus_book(data_args, tokenizer, model, image_size,
                                              padding_mode=padding_mode, split=split,)

    if data_args.modality in ['roc-aug', 'roc', 'book', 'yelp', 'commonGen', 'commonGen-aug'] and data_args.cache_mode=='no':
        dataset = TextDataset_NoCache(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
            model_emb=model
        )
    else:
        dataset = TextDataset(
            training_data,
            image_size,
            data_args,
            model_arch=data_args.model_arch,
        )



    if deterministic:

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=False,
            num_workers=1,
        )

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=True,
            num_workers=1,
        )
    while True:
        yield from data_loader

def helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, seqlen, data_args):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for (src_ids, input_ids) in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            tokenized_src = [vocab_dict.get(x, vocab_dict['UNK']) for x in src_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
            group_lst['src_ids'].append(tokenized_src)

        print(group_lst['word_ids'][:2])
        print('padding mode is pad')
        max_length = seqlen
        group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        print(max_src_length, seqlen)
        max_src_length = min(seqlen, max_src_length)
        group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                            vocab_dict['PAD'],
                                                                            max_src_length,
                                                                            return_mask=True)


        for input_ids, src_ids, src_mask in zip(group_lst['word_ids'], group_lst['src_ids'],
                                      group_lst['src_mask']):
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            result_train_lst.append({'input_ids': input_ids,
                                     'hidden_states': hidden_state.cpu().tolist(),
                                     'src_ids':src_ids,
                                     'src_mask':src_mask
                                     })

    return result_train_lst

def helper_tokenize_stream(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    import psutil
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    from datasets import Dataset as Dataset2
    raw_datasets = Dataset2.from_dict({'text':sentence_lst})
    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


    def tokenize_function(examples):
        if isinstance(vocab_dict, dict):
            input_ids = [[0] + [vocab_dict.get(x, vocab_dict['UNK']) for x in seq] + [1] for seq in examples['text']]
        elif isinstance(vocab_dict, PreTrainedTokenizerFast):
            examples['text'] = [" ".join(seq) for seq in examples['text']]
            input_ids = vocab_dict(examples['text'], add_special_tokens=True)['input_ids']
        result_dict = {'input_ids': input_ids}
        # clm input could be much much longer than block_size
        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print(tokenized_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    if padding_mode == 'block':
        block_size = seqlen
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result


        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    else:
        def pad_function(group_lst):
            max_length = seqlen
            if isinstance(vocab_dict, dict):
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict['PAD'], max_length)
            else:
                group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
            return group_lst

        # Process.memory_info is expressed in bytes, so convert to megabytes
        print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

        lm_datasets = tokenized_datasets.map(
            pad_function,
            batched=True,
            num_proc=1,
            desc=f"padding",
        )


    print(lm_datasets, 'padded dataset')
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    import datasets
    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets

def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for input_ids in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
        print(group_lst['word_ids'][:2])

        if padding_mode == 'block':
            print('padding mode is block')
            concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
            total_length = len(concatenated_examples[list(group_lst.keys())[0]])
            block_size = seqlen
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            group_lst = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        elif padding_mode == 'pad':
            print('padding mode is pad')
            max_length = seqlen
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)

        for input_ids in group_lst['word_ids']:
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            elif data_args.experiment == 'glove':
                hidden_state = model(torch.tensor(input_ids))
            result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})

    return result_train_lst



def get_corpus_rocstory(data_args, model, image_size, padding_mode='block',
                        split='train', load_vocab=None, subtask=None):
    import csv, torch, json
    from spacy.lang.en import English

    if data_args.experiment_mode == 'lm':
        if data_args.modality == 'roc':
            print('loading dataset from ROCStory')
            nlp = English()
            tokenizer = nlp.tokenizer
            sentence_lst = []
            print(f'loading from {data_args.roc_train}')
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.roc_train}/roc_train.json'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.roc_train}/roc_valid.json'
            else:
                assert False, "invalid split for ROC dataset"

            with open(path, 'r') as roc_reader:
                for row in roc_reader:
                    sentences = json.loads(row)[0].strip()
                    word_lst = [x.text for x in tokenizer(sentences)]
                    sentence_lst.append(word_lst)

            # with open(data_args.roc_train, 'r') as csvfile:
            #     roc_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
            #     for row in roc_reader:
            #         # tokenize.
            #         sentences = " ".join(row[2:])
            #         word_lst = [x.text for x in tokenizer(sentences)]
            #         sentence_lst.append(word_lst)
            # sentence_lst = sentence_lst[1:]
            print(sentence_lst[:2])

        if data_args.modality == 'roc-aug':
            print('loading dataset from ROCStory')
            nlp = English()
            tokenizer = nlp.tokenizer
            sentence_lst = []
            if split == 'train':
                print('loading form the TRAIN set')
                path_lst = [f'{data_args.roc_train}/roc_train.json']
                path_lst.append('diffusion_lm/improved-diffusion/diff_models/rocstories_gptj.txt')
                # path_lst.append('diffusion_lm/improved-diffusion/cache/ar_model_augment_roc.json')
                # path_lst.append('diffusion_lm/improved-diffusion/cache/ar_model_augment_roc2.json')

            elif split == 'valid':
                print('loading form the VALID set')
                path_lst = [f'{data_args.roc_train}/roc_valid.json']
            else:
                assert False, "invalid split for ROC dataset"

            print(path_lst)
            for path in path_lst:
                if path.endswith('txt'):
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = row.strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
                else:
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = json.loads(row)[0].strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2],sentence_lst[-2:], 'dataset size=',len(sentence_lst))
        elif data_args.modality == 'simple-wiki':
            print('loading dataset from simple wikipedia')
            sentence_lst = []
            with open(data_args.wiki_train, 'r') as ff:
                for row in ff:
                    word_lst = row.lower().split()
                    sentence_lst.append(word_lst)
            print(sentence_lst[:2])
        elif data_args.modality == 'e2e-tgt':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/src1_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/src1_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.e2e_train}/src1_test.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        word_lst = row.split('||')[1]
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)
            print(sentence_lst[:2])

        ####################################################################################################
        #                                        ADDED
        ####################################################################################################
        elif data_args.modality == "zuco":
            print(f'loading dataset from ZuCO {split}-{subtask} set')
            nlp = English()
            tokenizer = nlp.tokenizer
            sentence_lst = []
            path = f'{data_args.zuco_train}/{subtask}_{split}_0.8_0.1_0.1.json'
            print(os.path.exists(path), path)
            with open(path, 'r') as roc_reader:
                for row in roc_reader:
                    sentences = json.loads(row)[0].strip()
                    word_lst = [x.text for x in tokenizer(sentences)]
                    sentence_lst.append(word_lst)
        ####################################################################################################
        #                                        END ADDED
        ####################################################################################################
        
        elif data_args.modality == 'yelp':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.yelp_train}/yelpnlg-train.csv'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.yelp_train}/yelpnlg-dev.csv'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.yelp_train}/yelpnlg-test.csv'
            if split in ['train', 'valid', 'test']:

                with open(path, 'r') as csvfile:
                    yelp_reader = csv.reader(csvfile) #delimiter=' ', quotechar='|')
                    for row in yelp_reader:
                        sentences = row[1]
                        word_lst = [x.text for x in tokenizer(sentences)]
                        sentence_lst.append(word_lst)
                sentence_lst = sentence_lst[1:]
            print(sentence_lst[:2])

        elif data_args.modality == 'commonGen':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.commonGen_train}/commongen.train.jsonl'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.commonGen_train}/commongen.dev.jsonl'
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.commonGen_train}/commongen.test.jsonl'
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for line in ff:
                        line = json.loads(line)
                        for sentences in line['scene']:
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2])

        elif data_args.modality == 'commonGen-aug':
            print('loading dataset from simple YelpNLG dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.commonGen_train}/commongen.train.jsonl'
                path_lst = [f'{data_args.roc_train}/roc_train.json']
                path_lst.append('diffusion_lm/improved-diffusion/diff_models/rocstories_gptj.txt')
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.commonGen_train}/commongen.dev.jsonl'
                path_lst = []
            elif split == 'test':
                print('loading form the TEST set')
                path = f'{data_args.commonGen_train}/commongen.test.jsonl'
                path_lst = []

            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for line in ff:
                        line = json.loads(line)
                        for sentences in line['scene']:
                            word_lst = [x.text for x in tokenizer(sentences)]
                            sentence_lst.append(word_lst)
            print(sentence_lst[:2])
            import itertools
            for path in path_lst:
                if path.endswith('txt'):
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = row.strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            spl = [[]]
                            for x, y in itertools.groupby(word_lst, lambda z: z == '.'):
                                spl[-1].extend(y)
                                if x: spl.append([])
                            sentence_lst.extend(spl[:-1])
                else:
                    with open(path, 'r') as roc_reader:
                        for row in roc_reader:
                            sentences = json.loads(row)[0].strip()
                            word_lst = [x.text for x in tokenizer(sentences)]
                            spl = [[]]
                            for x, y in itertools.groupby(word_lst, lambda z: z == '.'):
                                spl[-1].extend(y)
                                if x: spl.append([])
                            sentence_lst.extend(spl[:-1])

            print(sentence_lst[-2:])


        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for input_ids in sentence_lst:
                counter.update(input_ids)

    if data_args.experiment_mode == 'conditional_gen':
        if data_args.modality == 'e2e':
            print('loading dataset from simple e2e dataset')
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                path = f'{data_args.e2e_train}/src1_train.txt'
                with open(path, 'r') as ff:
                    for row in ff:
                        src_lst, word_lst = row.split('||')
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        src_lst = [x.text for x in tokenizer(src_lst)]
                        sentence_lst.append((src_lst, word_lst))
            elif split == 'valid':
                path = f'{data_args.e2e_train}/src1_valid.txt'
                sentence_lst = read_e2e_files(path, data_args, tokenizer)
            print(sentence_lst[:2])
        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for (src_ids, input_ids) in sentence_lst:
                counter.update(input_ids)
                counter.update(src_ids)

    if load_vocab is None:
        vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3}
        for k, v in counter.items():
            if v > 10:
                vocab_dict[k] = len(vocab_dict)
        print(len(counter), len(vocab_dict))

        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        print(f'save the vocab to {path_save_vocab}')
        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        if not os.path.exists(path_save_vocab):
            print(f'save the vocab to {path_save_vocab}')
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(data_args.checkpoint_path)
            else:
                assert False, "invalid type of vocab_dict"



    if model is None and data_args.experiment == 'random':
        model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)
    elif data_args.experiment == 'gpt2_pre_compress':
        assert model is not None
    elif data_args.experiment == 'glove':
        assert data_args.in_channel == 50
        model = load_glove(vocab_dict)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)

    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    if not os.path.exists(path_save) and data_args.experiment == 'random':
        torch.save(model.state_dict(), path_save)


    if data_args.experiment_mode == 'lm' and data_args.modality in ['roc-aug', 'roc', 'yelp', 'commonGen', 'commonGen-aug'] \
            and data_args.cache_mode=='no':
        train_dataset = helper_tokenize_stream(sentence_lst, vocab_dict, model, image_size**2, data_args, padding_mode)
        return train_dataset, model
    elif data_args.experiment_mode == 'lm':
        result_train_lst = helper_tokenize_encode(sentence_lst, vocab_dict, model, image_size**2, data_args, padding_mode)
    elif data_args.experiment_mode == 'conditional_gen':
        result_train_lst = helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, image_size ** 2, data_args)
    return {'train': result_train_lst}, model
       


def get_all_zuco_v1_data(data_dir, taskname):

    input_mat_files_dir = os.path.join(data_dir, f"{taskname}", "Matlab_files")
    
    """load files"""
    mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
    mat_files = sorted(mat_files)

    if len(mat_files) == 0:
        print(f'No mat files found for {taskname}')
        quit()

    dataset_dict = {}   
    sentences = [] 
 
    for mat_file in tqdm(mat_files):
        
        """get subject id"""
        subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
        dataset_dict[subject_name] = []

        
        matdata = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']
        
        for sentence_data in matdata:
            word_level_data = sentence_data.word
            if not isinstance(word_level_data, float):
                # Get sentence level features
                sentence_obj = {'sentence_content': sentence_data.content}
                sentence_obj['sentence_level_EEG'] = {
                        'mean_t1':sentence_data.mean_t1_sec, 
                        'mean_t2':sentence_data.mean_t2_sec, 
                        'mean_a1':sentence_data.mean_a1_sec, 
                        'mean_a2':sentence_data.mean_a2_sec, 
                        'mean_b1':sentence_data.mean_b1_sec, 
                        'mean_b2':sentence_data.mean_b2_sec, 
                        'mean_g1':sentence_data.mean_g1_sec, 
                        'mean_g2':sentence_data.mean_g2_sec,
                    }
                sentence_obj['sentence_level_EEG_diff'] = {
                        'mean_t1_diff':sentence_data.mean_t1_diff_sec, 
                        'mean_t2_diff':sentence_data.mean_t2_diff_sec, 
                        'mean_a1_diff':sentence_data.mean_a1_diff_sec, 
                        'mean_a2_diff':sentence_data.mean_a2_diff_sec, 
                        'mean_b1_diff':sentence_data.mean_b1_diff_sec, 
                        'mean_b2_diff':sentence_data.mean_b2_diff_sec, 
                        'mean_g1_diff':sentence_data.mean_g1_diff_sec, 
                        'mean_g2_diff':sentence_data.mean_g2_diff_sec,
                    }
                
                #Get answers for Task1-NR
                if taskname == 'task1-SR':
                    sentence_obj['answer_EEG'] = {
                        'answer_mean_t1':sentence_data.answer_mean_t1_sec, 
                        'answer_mean_t2':sentence_data.answer_mean_t2_sec, 
                        'answer_mean_a1':sentence_data.answer_mean_a1_sec, 
                        'answer_mean_a2':sentence_data.answer_mean_a2_sec, 
                        'answer_mean_b1':sentence_data.answer_mean_b1_sec, 
                        'answer_mean_b2':sentence_data.answer_mean_b2_sec, 
                        'answer_mean_g1':sentence_data.answer_mean_g1_sec, 
                        'answer_mean_g2':sentence_data.answer_mean_g2_sec}
                    sentence_obj['answer_EEG_diff'] = {
                        'answer_mean_t1':sentence_data.answer_mean_t1_diff_sec, 
                        'answer_mean_t2':sentence_data.answer_mean_t2_diff_sec, 
                        'answer_mean_a1':sentence_data.answer_mean_a1_diff_sec, 
                        'answer_mean_a2':sentence_data.answer_mean_a2_diff_sec, 
                        'answer_mean_b1':sentence_data.answer_mean_b1_diff_sec, 
                        'answer_mean_b2':sentence_data.answer_mean_b2_diff_sec, 
                        'answer_mean_g1':sentence_data.answer_mean_g1_diff_sec, 
                        'answer_mean_g2':sentence_data.answer_mean_g2_diff_sec}
          
          


                # Get word level features
                sentence_obj['word'] = []
                sentence_obj['omissionRate'] = []
            
                word_tokens_has_fixation = [] 
                word_tokens_with_mask = []
                word_tokens_all = []

                for word in word_level_data:
                    word_obj = {'word_content':word.content}
                    word_tokens_all.append(word.content)
                    word_obj['nFixations'] = word.nFixations
                    if word.nFixations > 0:  
                        word_obj['word_level_EEG'] = {
                            'FFD': {
                                    'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 
                                    'FFD_a1':word.FFD_a1, 'FFD_a2':word.FFD_a2, 
                                    'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 
                                    'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2
                                   },
                            'TRT': {
                                    'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 
                                    'TRT_a1':word.TRT_a1, 'TRT_a2':word.TRT_a2, 
                                    'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 
                                    'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2
                                   },
                            'GD':  {
                                    'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 
                                    'GD_a1':word.GD_a1, 'GD_a2':word.GD_a2, 
                                    'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 
                                    'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2
                                   }      
                        }
                        sentence_obj['word'].append(word_obj)
                        word_tokens_has_fixation.append(word.content)
                        word_tokens_with_mask.append(word.content)
                    else:
                        word_tokens_with_mask.append('[MASK]')
                
                sentence_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sentence_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sentence_obj['word_tokens_all'] = word_tokens_all
            
                dataset_dict[subject_name].append(sentence_obj)

            else:
                print(f'missing sent: subj:{subject_name} content:{sentence_data.content}, return None')
                dataset_dict[subject_name].append(None)

                continue
    return dataset_dict









def get_all_zuco_v2_data(data_dir):

    input_mat_files_dir = os.path.join(data_dir, "task2-NR-2.0", "Matlab_files")
    
    mat_files = glob(os.path.join(input_mat_files_dir,'*.mat'))
    mat_files = sorted(mat_files)

    for mat_file in tqdm(mat_files):

        """get subject id"""
        subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
        dataset_dict = {}

        if subject_name !='YMH': # Remove this subject for dyslexy issue
            dataset_dict[subject_name] = []
            f = h5py.File(mat_file,'r')
            # print('keys in f:', list(f.keys()))
            sentence_data = f['sentenceData']
            # print('keys in sentence_data:', list(sentence_data.keys()))
            
            # sent level eeg 
            # mean_t1 = np.squeeze(f[sentence_data['mean_t1'][0][0]][()])
            mean_t1_objs = sentence_data['mean_t1']
            mean_t2_objs = sentence_data['mean_t2']
            mean_a1_objs = sentence_data['mean_a1']
            mean_a2_objs = sentence_data['mean_a2']
            mean_b1_objs = sentence_data['mean_b1']
            mean_b2_objs = sentence_data['mean_b2']
            mean_g1_objs = sentence_data['mean_g1']
            mean_g2_objs = sentence_data['mean_g2']
            
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            # print('contentData shape:', contentData.shape, 'dtype:', contentData.dtype)
            omissionR = sentence_data['omissionRate']
            wordData = sentence_data['word']


            for idx in range(len(rawData)):
                # get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = dh.load_matlab_string(f[obj_reference_content])
                # print('sentence string:', sent_string)
                
                sent_obj = {'content':sent_string}
                
                # get sentence level EEG
                sent_obj['sentence_level_EEG'] = {
                    'mean_t1':np.squeeze(f[mean_t1_objs[idx][0]][()]), 
                    'mean_t2':np.squeeze(f[mean_t2_objs[idx][0]][()]), 
                    'mean_a1':np.squeeze(f[mean_a1_objs[idx][0]][()]), 
                    'mean_a2':np.squeeze(f[mean_a2_objs[idx][0]][()]), 
                    'mean_b1':np.squeeze(f[mean_b1_objs[idx][0]][()]), 
                    'mean_b2':np.squeeze(f[mean_b2_objs[idx][0]][()]), 
                    'mean_g1':np.squeeze(f[mean_g1_objs[idx][0]][()]), 
                    'mean_g2':np.squeeze(f[mean_g2_objs[idx][0]][()])
                }
                # print(sent_obj)
                sent_obj['word'] = []

                # get word level data
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(f, f[wordData[idx][0]])
                
                if word_data == {}:
                    print(f'missing sent: subj:{subject_name} content:{sent_string}, append None')
                    dataset_dict[subject_name].append(None)
                    continue
                elif len(word_tokens_all) == 0:
                    print(f'no word level features: subj:{subject_name} content:{sent_string}, append None')
                    dataset_dict[subject_name].append(None)
                    continue

                else:                    
                    for widx in range(len(word_data)):
                        data_dict = word_data[widx]
                        word_obj = {'content':data_dict['content'], 'nFixations': data_dict['nFix']}
                        if 'GD_EEG' in data_dict:
                            # print('has fixation: ', data_dict['content'])
                            gd = data_dict["GD_EEG"]
                            ffd = data_dict["FFD_EEG"]
                            trt = data_dict["TRT_EEG"]
                            assert len(gd) == len(trt) == len(ffd) == 8
                            word_obj['word_level_EEG'] = {
                                'GD':{'GD_t1':gd[0], 'GD_t2':gd[1], 'GD_a1':gd[2], 'GD_a2':gd[3], 'GD_b1':gd[4], 'GD_b2':gd[5], 'GD_g1':gd[6], 'GD_g2':gd[7]},
                                'FFD':{'FFD_t1':ffd[0], 'FFD_t2':ffd[1], 'FFD_a1':ffd[2], 'FFD_a2':ffd[3], 'FFD_b1':ffd[4], 'FFD_b2':ffd[5], 'FFD_g1':ffd[6], 'FFD_g2':ffd[7]},
                                'TRT':{'TRT_t1':trt[0], 'TRT_t2':trt[1], 'TRT_a1':trt[2], 'TRT_a2':trt[3], 'TRT_b1':trt[4], 'TRT_b2':trt[5], 'TRT_g1':trt[6], 'TRT_g2':trt[7]}
                            }
                            sent_obj['word'].append(word_obj)
                        
                    sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                    sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                    sent_obj['word_tokens_all'] = word_tokens_all     
                    
                    # print(sent_obj.keys())
                    # print(len(sent_obj['word']))
                    # print(sent_obj['word'][0])

                    dataset_dict[subject].append(sent_obj)













def get_input_samples(input_dataset_dict):
    return

class ZucoDataset(Dataset):

    def __init__(self, 
                input_dataset_dicts,
                 tokenizer, 
                 subject = 'ALL', 
                 eeg_type = 'GD', 
                 bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                 data_config="unique_sentence"
                 ):
        """
        args:
            :param: input_dataset_dicts 

        """
        self.eeg_type = eeg_type
        self.bands = bands
        self.tokenizer = tokenizer
        self.eeg_data = []
        self.sentence_data = []
        self.subjects = subject

        if data_config=="unique_sentence":
            # We create datasets with different sentence in train and validation
            pass
        elif data_config == 'unique_subj':
            # We split the data over the subjects
            pass


        total_num_sentence = len(input_dataset_dict[subjects[0]])
            
        train_divider = int(0.8*total_num_sentence)
        dev_divider = train_divider + int(0.1*total_num_sentence)
        
        print(f'train divider = {train_divider}')
        print(f'dev divider = {dev_divider}')