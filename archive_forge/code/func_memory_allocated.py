import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def memory_allocated(device: Union[Device, int]=None) -> int:
    """Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    .. note::
        This is likely less than the amount shown in `nvidia-smi` since some
        unused memory can be held by the caching allocator and some context
        needs to be created on GPU. See :ref:`cuda-memory-management` for more
        details about GPU memory management.
    """
    return memory_stats(device=device).get('allocated_bytes.all.current', 0)