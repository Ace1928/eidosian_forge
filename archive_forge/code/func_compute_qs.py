from collections import defaultdict
import logging
import math
from typing import Dict
import torch
import torch.distributed as dist
from . import default_hooks as default
from torch.distributed import distributed_c10d
def compute_qs(fut):
    state.p_memory_dict[bucket_index] = fut.value()
    for p in ps:
        _orthogonalize(p, state.orthogonalization_epsilon)
    for tensor, p, q in zip(tensors_to_compress, ps, qs):
        torch.bmm(tensor.transpose(1, 2), p, out=q)
    return dist.all_reduce(state.q_memory_dict[bucket_index], group=group_to_use, async_op=True).get_future().wait()[0]