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
def multi_margin_loss(input: Tensor, target: Tensor, p: int=1, margin: float=1.0, weight: Optional[Tensor]=None, size_average: Optional[bool]=None, reduce: Optional[bool]=None, reduction: str='mean') -> Tensor:
    """multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    if has_torch_function_variadic(input, target, weight):
        return handle_torch_function(multi_margin_loss, (input, target, weight), input, target, p=p, margin=margin, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)
    if p != 1 and p != 2:
        raise ValueError('only p == 1 and p == 2 supported')
    if weight is not None:
        if weight.dim() != 1:
            raise ValueError('weight must be one-dimensional')
    return torch._C._nn.multi_margin_loss(input, target, p, margin, weight, reduction_enum)