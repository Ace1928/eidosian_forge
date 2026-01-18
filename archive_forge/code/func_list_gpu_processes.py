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
def list_gpu_processes(device: Union[Device, int]=None) -> str:
    """Return a human-readable printout of the running processes and their GPU memory use for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (torch.device or int, optional): selected device. Returns
            printout for the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    try:
        import pynvml
    except ModuleNotFoundError:
        return 'pynvml module not found, please install pynvml'
    from pynvml import NVMLError_DriverNotLoaded
    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        return "cuda driver can't be loaded, is cuda enabled?"
    device = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    lines = []
    lines.append(f'GPU:{device}')
    if len(procs) == 0:
        lines.append('no processes are running')
    for p in procs:
        mem = p.usedGpuMemory / (1024 * 1024)
        lines.append(f'process {p.pid:>10d} uses {mem:>12.3f} MB GPU memory')
    return '\n'.join(lines)