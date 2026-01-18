import torch
from .. import Tensor
from . import profiler
from .event import Event
def current_allocated_memory() -> int:
    """Returns the current GPU memory occupied by tensors in bytes.

    .. note::
       The returned size does not include cached allocations in
       memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()