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
def is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Check if `torch_xla` is available. To train a native pytorch job in an environment with torch xla installed, set
    the USE_TORCH_XLA to false.
    """
    assert not (check_is_tpu and check_is_gpu), 'The check_is_tpu and check_is_gpu cannot both be true.'
    if not _torch_xla_available:
        return False
    elif check_is_gpu:
        return torch_xla.runtime.device_type() in ['GPU', 'CUDA']
    elif check_is_tpu:
        return torch_xla.runtime.device_type() == 'TPU'
    return True