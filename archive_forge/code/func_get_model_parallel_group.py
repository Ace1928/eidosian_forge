from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_model_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, 'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP