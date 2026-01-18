import functools
import os
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, List
from . import language as tl
from ._C.libtriton.triton import runtime
def get_dram_gbps(backend=None, device=None):
    """ return DRAM bandwidth in GB/s """
    import torch
    from .runtime import driver
    if not backend:
        backend = runtime.backend.CUDA
    if not device:
        device = torch.cuda.current_device()
    mem_clock_khz = driver.utils.get_device_properties(device)['mem_clock_rate']
    bus_width = driver.utils.get_device_properties(device)['mem_bus_width']
    bw_gbps = mem_clock_khz * bus_width * 2 / 1000000.0 / 8
    return bw_gbps