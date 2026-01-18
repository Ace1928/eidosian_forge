import builtins
import collections
import inspect
import itertools
import math
import operator
import warnings
from collections.abc import Iterable
from enum import Enum
from functools import partial, reduce, singledispatch, wraps
from typing import Any, Callable, Dict, List, Optional, overload, Sequence, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
from torch import sym_float, sym_int
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._decomp import register_decomposition
import torch._refs._conversions
import torch._refs.fft
import torch._refs.linalg
import torch._refs.nn.functional
import torch._refs.special
@register_decomposition(aten.renorm)
@out_wrapper()
def renorm(input: TensorLikeType, p: RealNumberType, dim: int, maxnorm: RealNumberType) -> TensorLikeType:
    torch._check(not isinstance(p, complex), lambda: 'renorm: p must be real-valued')
    torch._check(p > 0, lambda: 'renorm: non-positive norm not supported')
    torch._check(not isinstance(maxnorm, complex), lambda: 'renorm: maxnorm must be real-valued')
    torch._check(maxnorm >= 0, lambda: f'renorm: expected maxnorm to be >= 0 but got {maxnorm}')
    ndim = input.ndim
    torch._check(ndim > 1, lambda: f'renorm: input needs at least 2 dimensions, got {ndim} dimensions')
    dim = utils.canonicalize_dim(ndim, dim)
    reduce_dims = list(range(ndim))
    del reduce_dims[dim]
    acc_type = utils.get_computation_dtype(input.dtype)
    if acc_type != input.dtype:
        norm = torch.linalg.vector_norm(input, p, reduce_dims, keepdim=True, dtype=acc_type)
    else:
        norm = torch.linalg.vector_norm(input, p, reduce_dims, keepdim=True)
    eps = 1e-07
    norm_factor = torch.where(norm > maxnorm, maxnorm / (norm + eps), 1.0)
    if acc_type != input.dtype:
        norm_factor = prims.convert_element_type(norm_factor, input.dtype)
    return (input * norm_factor).contiguous()