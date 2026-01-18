from __future__ import annotations
from typing import Optional
import torch
from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util
from ._normalizations import (
def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(x1: ArrayLikeOrScalar, x2: ArrayLikeOrScalar, /, out: Optional[OutArray]=None, *, where: NotImplementedType=True, casting: Optional[CastingModes]='same_kind', order: NotImplementedType='K', dtype: Optional[DTypeLike]=None, subok: NotImplementedType=False, signature: NotImplementedType=None, extobj: NotImplementedType=None):
        if dtype is not None:

            def cast(x, dtype):
                if isinstance(x, torch.Tensor):
                    return _util.typecast_tensor(x, dtype, casting)
                else:
                    return torch.as_tensor(x, dtype=dtype)
            x1 = cast(x1, dtype)
            x2 = cast(x2, dtype)
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            dtype = _dtypes_impl.result_type_impl(x1, x2)
            x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)
        else:
            x1, x2 = _dtypes_impl.nep50_to_tensors(x1, x2, torch_func.__name__ in NEP50_FUNCS, torch_func.__name__)
        result = torch_func(x1, x2)
        return _ufunc_postprocess(result, out, casting)
    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__
    return wrapped