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
def gather_across_data_parallel_groups(tensor):
    """
    Recursively gather tensor in a nested list/tuple/dictionary of tensors from data parallel ranks.

    Args:
        tensor (nested list/tuple/dictionary of `torch.Tensor`):
            The data to gather across data parallel ranks.

    """

    def _gpu_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        output_tensors = [torch.empty_like(tensor) for _ in range(torch.distributed.get_world_size(group=mpu.get_data_parallel_group()))]
        torch.distributed.all_gather(output_tensors, tensor, group=mpu.get_data_parallel_group())
        return torch.cat(output_tensors, dim=0)
    return recursively_apply(_gpu_gather_one, tensor, error_on_other_type=True)