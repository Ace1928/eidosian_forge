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
def get_paged(*shape, dtype=torch.float32, device=FIRST_CUDA_DEVICE):
    num_bytes = dtype2bytes[dtype] * prod(shape)
    cuda_ptr = lib.cget_managed_ptr(ct.c_size_t(num_bytes))
    c_ptr = ct.cast(cuda_ptr, ct.POINTER(ct.c_int))
    new_array = np.ctypeslib.as_array(c_ptr, shape=shape)
    out = torch.frombuffer(new_array, dtype=dtype, count=prod(shape)).view(shape)
    out.is_paged = True
    out.page_deviceid = device.index
    return out