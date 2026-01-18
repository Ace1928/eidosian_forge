from typing import Optional
import torch
from torch import Tensor
from torch.distributed import ProcessGroup
def get_dim_for_local_rank(dim: int, world_size: int, local_rank: int, multiple_of: int=1) -> int:
    """Get the dim for the local rank derived from splitting dim on world_size processes.

    The split may not be even across the world_size processes.
    """
    multiple = dim // multiple_of
    div = multiple // world_size
    mod = multiple % world_size
    local_multiple = div + int(local_rank < mod)
    return local_multiple * multiple_of