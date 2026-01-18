import functools
import importlib
import logging
import os
import tempfile
import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend
@functools.lru_cache(None)
def llvm_target():
    if 'avx512' in open('/proc/cpuinfo').read():
        return 'llvm -mcpu=skylake-avx512'
    return 'llvm -mcpu=core-avx2'