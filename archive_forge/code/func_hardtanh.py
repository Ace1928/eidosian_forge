from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def hardtanh(input: Tensor, min_val: float=-1.0, max_val: float=1.0, inplace: bool=False) -> Tensor:
    """This is the quantized version of :func:`~torch.nn.functional.hardtanh`.
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardtanh' must be quantized!")
    if inplace:
        return torch._C._nn.hardtanh_(input, min_val, max_val)
    return torch._C._nn.hardtanh(input, min_val, max_val)