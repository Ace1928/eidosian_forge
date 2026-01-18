from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def initial_seed() -> int:
    """Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    _lazy_init()
    idx = current_device()
    default_generator = torch.cuda.default_generators[idx]
    return default_generator.initial_seed()