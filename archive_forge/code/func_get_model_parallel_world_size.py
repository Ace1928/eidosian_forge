from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_model_parallel_world_size() -> int:
    """Return world size for the model parallel group."""
    return torch.distributed.get_world_size(group=get_model_parallel_group())