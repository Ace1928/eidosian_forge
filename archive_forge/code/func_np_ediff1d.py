import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@overload(np.ediff1d)
def np_ediff1d(ary, to_end=None, to_begin=None):
    if isinstance(ary, types.Array):
        if isinstance(ary.dtype, types.Boolean):
            raise NumbaTypeError('Boolean dtype is unsupported (as per NumPy)')
    ary_dt = _dtype_of_compound(ary)
    to_begin_dt = None
    if not is_nonelike(to_begin):
        to_begin_dt = _dtype_of_compound(to_begin)
    to_end_dt = None
    if not is_nonelike(to_end):
        to_end_dt = _dtype_of_compound(to_end)
    if to_begin_dt is not None and (not np.can_cast(to_begin_dt, ary_dt)):
        msg = 'dtype of to_begin must be compatible with input ary'
        raise NumbaTypeError(msg)
    if to_end_dt is not None and (not np.can_cast(to_end_dt, ary_dt)):
        msg = 'dtype of to_end must be compatible with input ary'
        raise NumbaTypeError(msg)

    def np_ediff1d_impl(ary, to_end=None, to_begin=None):
        start = _prepare_array(to_begin)
        mid = _prepare_array(ary)
        end = _prepare_array(to_end)
        out_dtype = mid.dtype
        if len(mid) > 0:
            out = np.empty(len(start) + len(mid) + len(end) - 1, dtype=out_dtype)
            start_idx = len(start)
            mid_idx = len(start) + len(mid) - 1
            out[:start_idx] = start
            out[start_idx:mid_idx] = np.diff(mid)
            out[mid_idx:] = end
        else:
            out = np.empty(len(start) + len(end), dtype=out_dtype)
            start_idx = len(start)
            out[:start_idx] = start
            out[start_idx:] = end
        return out
    return np_ediff1d_impl