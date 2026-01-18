from typing import Iterable, List, Union
import torch
from .. import Tensor
from . import _lazy_call, _lazy_init, current_device, device_count
def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    """Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)