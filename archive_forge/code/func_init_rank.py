import collections
import warnings
from typing import Optional, Sequence, Union
import torch.cuda
def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)