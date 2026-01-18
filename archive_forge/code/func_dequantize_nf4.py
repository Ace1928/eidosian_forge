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
def dequantize_nf4(A: Tensor, quant_state: Optional[QuantState]=None, absmax: Optional[torch.Tensor]=None, out: Optional[torch.Tensor]=None, blocksize: int=64) -> Tensor:
    return dequantize_4bit(A, quant_state, absmax, out, blocksize, 'nf4')