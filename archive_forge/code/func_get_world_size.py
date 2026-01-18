import contextlib
import torch
from torch.distributed import ReduceOp
def get_world_size() -> int:
    """Returns the world size."""
    return _WORLD_SIZE