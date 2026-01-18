import os
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator, List, Optional, Union, cast
import torch
from typing_extensions import override
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.accelerators.registry import _AcceleratorRegistry
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_info
@lru_cache(1)
def num_cuda_devices() -> int:
    """Returns the number of available CUDA devices.

    Unlike :func:`torch.cuda.device_count`, this function does its best not to create a CUDA context for fork support,
    if the platform allows it.

    """
    if _TORCH_GREATER_EQUAL_2_0:
        return torch.cuda.device_count()
    nvml_count = _device_count_nvml()
    return torch.cuda.device_count() if nvml_count < 0 else nvml_count