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
def is_mlu_available(check_device=False):
    """Checks if `torch_mlu` is installed and potentially if a MLU is in the environment"""
    if importlib.util.find_spec('torch_mlu') is None:
        return False
    import torch
    import torch_mlu
    if check_device:
        try:
            _ = torch.mlu.device_count()
            return torch.mlu.is_available()
        except RuntimeError:
            return False
    return hasattr(torch, 'mlu') and torch.mlu.is_available()