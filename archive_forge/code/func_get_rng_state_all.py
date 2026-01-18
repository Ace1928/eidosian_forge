from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def get_rng_state_all() -> List[Tensor]:
    """Return a list of ByteTensor representing the random number states of all devices."""
    results = []
    for i in range(device_count()):
        results.append(get_rng_state(i))
    return results