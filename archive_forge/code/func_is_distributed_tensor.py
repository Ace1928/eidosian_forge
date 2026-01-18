import torch
from torch.utils import _pytree as pytree
from typing import Optional
def is_distributed_tensor(e):
    nonlocal has_distributed_tensor
    if isinstance(e, ShardedTensor):
        has_distributed_tensor = True