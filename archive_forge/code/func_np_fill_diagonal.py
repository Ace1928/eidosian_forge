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
@overload(np.fill_diagonal)
def np_fill_diagonal(a, val, wrap=False):
    if a.ndim > 1:
        if isinstance(a.dtype, types.Integer):
            checker = _check_val_int
        elif isinstance(a.dtype, types.Float):
            checker = _check_val_float
        else:
            checker = _check_nop

        def scalar_impl(a, val, wrap=False):
            tmpval = _asarray(val).flatten()
            checker(a, tmpval)
            _fill_diagonal_scalar(a, val, wrap)

        def non_scalar_impl(a, val, wrap=False):
            tmpval = _asarray(val).flatten()
            checker(a, tmpval)
            _fill_diagonal(a, tmpval, wrap)
        if isinstance(val, (types.Float, types.Integer, types.Boolean)):
            return scalar_impl
        elif isinstance(val, (types.Tuple, types.Sequence, types.Array)):
            return non_scalar_impl
    else:
        msg = 'The first argument must be at least 2-D (found %s-D)' % a.ndim
        raise TypingError(msg)