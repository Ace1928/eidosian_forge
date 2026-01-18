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
def is_on_gpu(tensors):
    on_gpu = True
    gpu_ids = set()
    for t in tensors:
        if t is None:
            continue
        is_paged = getattr(t, 'is_paged', False)
        on_gpu &= t.device.type == 'cuda' or is_paged
        if not is_paged:
            gpu_ids.add(t.device.index)
    if not on_gpu:
        raise TypeError(f'All input tensors need to be on the same GPU, but found some tensors to not be on a GPU:\n {[(t.shape, t.device) for t in tensors]}')
    if len(gpu_ids) > 1:
        raise TypeError(f'Input tensors need to be on the same GPU, but found the following tensor and device combinations:\n {[(t.shape, t.device) for t in tensors]}')
    return on_gpu