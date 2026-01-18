from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def seed_all() -> None:
    """Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = torch.cuda.default_generators[i]
            if not seeded:
                default_generator.seed()
                random_seed = default_generator.initial_seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)
    _lazy_call(cb)