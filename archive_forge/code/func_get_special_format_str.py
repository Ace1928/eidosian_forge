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
def get_special_format_str():
    if not torch.cuda.is_available():
        return 'col_turing'
    major, _minor = torch.cuda.get_device_capability()
    if major <= 7:
        return 'col_turing'
    if major == 8:
        return 'col_ampere'
    return 'col_turing'