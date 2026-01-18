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
def build_train_valid_test_data_iterators(self):

    def cyclic_iter(iter):
        while True:
            yield from iter
    args = get_args()
    train_dataloader, valid_dataloader, test_dataloader = (None, None, None)
    print_rank_0('> building train, validation, and test datasets ...')
    if args.iteration > 0 and args.consumed_train_samples == 0:
        assert args.train_samples is None, 'only backward compatiblity support for iteration-based training'
        args.consumed_train_samples = args.iteration * args.global_batch_size
    if args.iteration > 0 and args.consumed_valid_samples == 0:
        if args.train_samples is None:
            args.consumed_valid_samples = args.iteration // args.eval_interval * args.eval_iters * args.global_batch_size
    if mpu.get_tensor_model_parallel_rank() == 0:
        if args.train_samples:
            train_samples = args.train_samples
        else:
            train_samples = args.train_iters * args.global_batch_size
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        test_iters = args.eval_iters
        train_val_test_num_samples = [train_samples, eval_iters * args.global_batch_size, test_iters * args.global_batch_size]
        print_rank_0(' > datasets target sizes (minimum size):')
        print_rank_0(f'    train:      {train_val_test_num_samples[0]}')
        print_rank_0(f'    validation: {train_val_test_num_samples[1]}')
        print_rank_0(f'    test:       {train_val_test_num_samples[2]}')
        train_valid_test_datasets_provider = self.get_train_valid_test_datasets_provider()
        train_ds, valid_ds, test_ds = train_valid_test_datasets_provider(train_val_test_num_samples)
        train_dataloader = self.build_pretraining_data_loader(train_ds, args.consumed_train_samples)
        valid_dataloader = self.build_pretraining_data_loader(valid_ds, args.consumed_valid_samples)
        test_dataloader = self.build_pretraining_data_loader(test_ds, 0)
        do_train = train_dataloader is not None and args.train_iters > 0
        do_valid = valid_dataloader is not None and args.eval_iters > 0
        do_test = test_dataloader is not None and args.eval_iters > 0
        flags = torch.cuda.LongTensor([int(do_train), int(do_valid), int(do_test)])
    else:
        flags = torch.cuda.LongTensor([0, 0, 0])
    torch.distributed.broadcast(flags, mpu.get_tensor_model_parallel_src_rank(), group=mpu.get_tensor_model_parallel_group())
    args.do_train = flags[0].item()
    args.do_valid = flags[1].item()
    args.do_test = flags[2].item()
    dl_type = args.dataloader_type
    assert dl_type in ['single', 'cyclic']
    if train_dataloader is not None:
        train_data_iterator = iter(train_dataloader) if dl_type == 'single' else iter(cyclic_iter(train_dataloader))
    else:
        train_data_iterator = None
    if valid_dataloader is not None:
        valid_data_iterator = iter(valid_dataloader) if dl_type == 'single' else iter(cyclic_iter(valid_dataloader))
    else:
        valid_data_iterator = None
    if test_dataloader is not None:
        test_data_iterator = iter(test_dataloader) if dl_type == 'single' else iter(cyclic_iter(test_dataloader))
    else:
        test_data_iterator = None
    return (train_data_iterator, valid_data_iterator, test_data_iterator)