from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_model_parallel_src_rank() -> int:
    """Calculate the global rank corresponding to a local rank zero
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return global_rank // local_world_size * local_world_size