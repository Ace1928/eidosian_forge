from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_data_parallel_rank() -> int:
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())