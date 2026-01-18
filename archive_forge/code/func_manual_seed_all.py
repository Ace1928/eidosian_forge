from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def manual_seed_all(seed: int) -> None:
    """Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(seed)
    _lazy_call(cb, seed_all=True)