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
def is_cuda_available():
    """
    Checks if `cuda` is available via an `nvml-based` check which won't trigger the drivers and leave cuda
    uninitialized.
    """
    pytorch_nvml_based_cuda_check_previous_value = os.environ.get('PYTORCH_NVML_BASED_CUDA_CHECK')
    try:
        os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = str(1)
        available = torch.cuda.is_available()
    finally:
        if pytorch_nvml_based_cuda_check_previous_value:
            os.environ['PYTORCH_NVML_BASED_CUDA_CHECK'] = pytorch_nvml_based_cuda_check_previous_value
        else:
            os.environ.pop('PYTORCH_NVML_BASED_CUDA_CHECK', None)
    return available