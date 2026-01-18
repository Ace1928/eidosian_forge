import logging
import math
import os
from copy import deepcopy
import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from accelerate import Accelerator, DataLoaderConfiguration, DistributedType
from accelerate.data_loader import DataLoaderDispatcher
from accelerate.test_utils import RegressionDataset, RegressionModel, torch_device
from accelerate.utils import is_torch_xla_available, set_seed
def get_dataloader(accelerator: Accelerator, use_longest=False):
    tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/mrpc-bert-base-cased')
    dataset = load_dataset('glue', 'mrpc', split='validation')

    def tokenize_function(examples):
        outputs = tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=None)
        return outputs
    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['idx', 'sentence1', 'sentence2'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')

    def collate_fn(examples):
        if use_longest:
            return tokenizer.pad(examples, padding='longest', return_tensors='pt')
        return tokenizer.pad(examples, padding='max_length', max_length=128, return_tensors='pt')
    return DataLoader(tokenized_datasets, shuffle=False, collate_fn=collate_fn, batch_size=16)