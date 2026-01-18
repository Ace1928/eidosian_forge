from collections import OrderedDict
import copy
import io
from itertools import chain
import logging
from math import inf
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.autograd import profiler
import torch.distributed as dist
from torch.nn import Parameter
from torch.optim import SGD, Optimizer
from fairscale.internal.params import calc_grad_norm, get_global_rank, recursive_copy_to_device
from fairscale.nn.misc import ParamBucket
def partition_parameters(self) -> List[List[dict]]:
    """Partitions parameters across distributed data parallel ranks.

        Returns a list of param_groups (which is a list of dict) where each
        element of the list contains the param_groups for a rank. Element 0
        corresponds to rank 0, etc. We need all the ranks for the broadcast
        inside step().
        """
    if len(self._partition_parameters) == 0:
        self._partition_parameters = [list() for _ in range(self.world_size)]
        sizes = [0] * self.world_size
        for param_group in self.param_groups:
            param_lists: List[List] = [list() for _ in range(self.world_size)]
            for param in param_group['params']:
                rank = sizes.index(min(sizes))
                param_lists[rank].append(param)
                if param.requires_grad:
                    sizes[rank] += param.numel()
                else:
                    sizes[rank] += 1
            for rank, params in enumerate(param_lists):
                param_group_rank = copy.copy(param_group)
                param_group_rank['params'] = params
                self._partition_parameters[rank].append(param_group_rank)
    return self._partition_parameters