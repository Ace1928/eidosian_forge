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
def is_npu_available(check_device=False):
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if importlib.util.find_spec('torch') is None or importlib.util.find_spec('torch_npu') is None:
        return False
    import torch
    import torch_npu
    if check_device:
        try:
            _ = torch.npu.device_count()
            return torch.npu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'npu') and torch.npu.is_available()