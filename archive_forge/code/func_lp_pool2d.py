from typing import Callable, List, Optional, Tuple, Union
import math
import warnings
import importlib
import torch
from torch import _VF
from torch import sym_int as _sym_int
from torch._C import _infer_size, _add_docstr
from torch._torch_docs import reproducibility_notes, tf32_notes, sparse_support_notes
from typing import TYPE_CHECKING
from .._jit_internal import boolean_dispatch, _overload, BroadcastingList1, BroadcastingList2, BroadcastingList3
from ..overrides import (
from . import _reduction as _Reduction
from . import grad  # noqa: F401
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
def lp_pool2d(input: Tensor, norm_type: Union[int, float], kernel_size: BroadcastingList2[int], stride: Optional[BroadcastingList2[int]]=None, ceil_mode: bool=False) -> Tensor:
    """
    Apply a 2D power-average pooling over an input signal composed of several input planes.

    If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(lp_pool2d, (input,), input, norm_type, kernel_size, stride=stride, ceil_mode=ceil_mode)
    kw, kh = utils._pair(kernel_size)
    if stride is not None:
        out = avg_pool2d(input.pow(norm_type), kernel_size, stride, 0, ceil_mode)
    else:
        out = avg_pool2d(input.pow(norm_type), kernel_size, padding=0, ceil_mode=ceil_mode)
    return (torch.sign(out) * relu(torch.abs(out))).mul(kw * kh).pow(1.0 / norm_type)