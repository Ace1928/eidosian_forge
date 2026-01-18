import torch
from .. import Tensor
from . import profiler
from .event import Event
def driver_allocated_memory() -> int:
    """Returns total GPU memory allocated by Metal driver for the process in bytes.

    .. note::
       The returned size includes cached allocations in MPSAllocator pools
       as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()