import argparse
import math
from abc import ABC
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from ..optimizer import AcceleratedOptimizer
from ..scheduler import AcceleratedScheduler
from .imports import is_megatron_lm_available, is_transformers_available
from .operations import recursively_apply, send_to_device
def get_train_valid_test_datasets_provider(self):

    def train_valid_test_datasets_provider(train_val_test_num_samples):
        """Build train, valid, and test datasets."""
        args = get_args()
        dataset_args = {'data_prefix': args.data_path, 'data_impl': args.data_impl, 'splits_string': args.split, 'train_valid_test_num_samples': train_val_test_num_samples, 'skip_warmup': not args.mmap_warmup, 'seed': args.seed}
        if args.model_type_name == 'bert':
            dataset_args.update({'max_seq_length': args.seq_length, 'masked_lm_prob': args.mask_prob, 'short_seq_prob': args.short_seq_prob, 'binary_head': args.bert_binary_head})
        elif args.model_type_name == 'gpt':
            dataset_args.update({'seq_length': args.seq_length})
        elif args.model_type_name == 't5':
            dataset_args.update({'max_seq_length': args.encoder_seq_length, 'max_seq_length_dec': args.decoder_seq_length, 'masked_lm_prob': args.mask_prob, 'short_seq_prob': args.short_seq_prob, 'dataset_type': 't5'})
        else:
            raise ValueError(f'Unsupported model type: {args.model_type_name}')
        if args.model_type_name == 'gpt':
            from megatron.data.gpt_dataset import build_train_valid_test_datasets
        else:
            from megatron.data.dataset_utils import build_train_valid_test_datasets
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(**dataset_args)
        return (train_ds, valid_ds, test_ds)
    return train_valid_test_datasets_provider