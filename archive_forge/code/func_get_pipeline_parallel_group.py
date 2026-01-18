from typing import List, Optional
import torch
from .utils import ensure_divisibility
def get_pipeline_parallel_group() -> torch.distributed.ProcessGroup:
    """Get the pipeline parallel group the caller rank belongs to."""
    assert _PIPELINE_PARALLEL_GROUP is not None, 'pipeline parallel group is not initialized'
    return _PIPELINE_PARALLEL_GROUP