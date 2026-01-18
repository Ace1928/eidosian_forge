import contextlib
from typing import Dict, Iterator, Set, Union
import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from .initialize import get_data_parallel_rank, get_model_parallel_rank
def get_cuda_rng_tracker() -> CudaRNGStatesTracker:
    """Get cuda rng tracker."""
    return _CUDA_RNG_STATE_TRACKER