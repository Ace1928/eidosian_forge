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
def max_unpool1d(input: Tensor, indices: Tensor, kernel_size: BroadcastingList1[int], stride: Optional[BroadcastingList1[int]]=None, padding: BroadcastingList1[int]=0, output_size: Optional[BroadcastingList1[int]]=None) -> Tensor:
    """Compute a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(max_unpool1d, (input,), input, indices, kernel_size, stride=stride, padding=padding, output_size=output_size)
    kernel_size = _single(kernel_size)
    if stride is not None:
        _stride = _single(stride)
    else:
        _stride = kernel_size
    padding = _single(padding)
    output_size = _unpool_output_size(input, kernel_size, _stride, padding, output_size)
    if isinstance(output_size, list):
        output_size = output_size + [1]
    else:
        output_size = output_size + (1,)
    return torch._C._nn.max_unpool2d(input.unsqueeze(-1), indices.unsqueeze(-1), output_size).squeeze(-1)