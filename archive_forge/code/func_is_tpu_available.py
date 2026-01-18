import importlib
import importlib.metadata
import os
import warnings
from functools import lru_cache
import torch
from packaging import version
from packaging.version import parse
from .environment import parse_flag_from_env, str_to_bool
from .versions import compare_versions, is_torch_version
@lru_cache
def is_tpu_available(check_device=True):
    """Checks if `torch_xla` is installed and potentially if a TPU is in the environment"""
    warnings.warn('`is_tpu_available` is deprecated and will be removed in v0.27.0. Please use the `is_torch_xla_available` instead.', FutureWarning)
    if is_cuda_available():
        return False
    if check_device:
        if _tpu_available:
            try:
                _ = xm.xla_device()
                return True
            except RuntimeError:
                return False
    return _tpu_available