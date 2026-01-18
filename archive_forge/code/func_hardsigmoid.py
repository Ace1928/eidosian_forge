from typing import List, Optional
import warnings
import torch
from torch import Tensor
from torch.nn.modules.utils import _pair, _triple
from torch.jit.annotations import BroadcastingList2
from .modules.utils import _pair_from_first
def hardsigmoid(input: Tensor, inplace: bool=False) -> Tensor:
    """This is the quantized version of :func:`~torch.nn.functional.hardsigmoid`.
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardsigmoid' must be quantized!")
    if inplace:
        return torch._C._nn.hardsigmoid_(input)
    return torch._C._nn.hardsigmoid(input)