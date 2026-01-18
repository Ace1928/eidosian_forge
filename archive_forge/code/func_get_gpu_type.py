import math
from enum import IntEnum
from typing import TYPE_CHECKING
import torch
from . import ir
from .utils import get_dtype_size, sympy_product
from .virtualized import V
def get_gpu_type() -> NVIDIA_GPU_TYPE:
    gpu_info = torch.utils.collect_env.get_gpu_info(torch.utils.collect_env.run)
    if 'V100' in gpu_info:
        return NVIDIA_GPU_TYPE.VOLTA
    elif 'A100' in gpu_info:
        return NVIDIA_GPU_TYPE.AMPERE
    elif 'H100' in gpu_info:
        return NVIDIA_GPU_TYPE.HOPPER
    else:
        return NVIDIA_GPU_TYPE.AMPERE