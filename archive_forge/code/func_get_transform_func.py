import ctypes as ct
from functools import reduce  # Required in Python 3
import itertools
import operator
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict
from .cextension import COMPILED_WITH_CUDA, lib
def get_transform_func(dtype, orderA, orderOut, transpose=False):
    name = f'ctransform_{(8 if dtype == torch.int8 else 32)}_{orderA}_to_{orderOut}_{('t' if transpose else 'n')}'
    if not hasattr(lib, name):
        print(name)
        raise ValueError(f'Transform function not supported: {orderA} to {orderOut} for data type {dtype} and transpose={transpose}')
    else:
        return getattr(lib, name)