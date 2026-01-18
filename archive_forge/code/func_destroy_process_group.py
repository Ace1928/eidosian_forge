import contextlib
import torch
from torch.distributed import ReduceOp
def destroy_process_group() -> None:
    """Destroys the NCCL backend."""
    global _NCCL_BACKEND
    global _WORLD_SIZE
    _NCCL_BACKEND = None
    _WORLD_SIZE = 0