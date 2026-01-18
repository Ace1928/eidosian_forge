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
def build_pretraining_data_loader(self, dataset, consumed_samples):
    if dataset is None:
        return None
    args = get_args()
    micro_batch_size = args.micro_batch_size * args.num_micro_batches
    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(total_samples=len(dataset), consumed_samples=consumed_samples, micro_batch_size=micro_batch_size, data_parallel_rank=mpu.get_data_parallel_rank(), data_parallel_size=mpu.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(dataset, total_samples=len(dataset), consumed_samples=consumed_samples, micro_batch_size=micro_batch_size, data_parallel_rank=mpu.get_data_parallel_rank(), data_parallel_size=mpu.get_data_parallel_world_size(), data_sharding=args.data_sharding)
    else:
        raise Exception(f'{args.dataloader_type} dataloader type is not supported.')
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)