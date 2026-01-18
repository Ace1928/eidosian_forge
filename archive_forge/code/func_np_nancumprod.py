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
@overload(np.nancumprod)
def np_nancumprod(a):
    if not isinstance(a, types.Array):
        return
    if isinstance(a.dtype, (types.Boolean, types.Integer)):
        return lambda a: np.cumprod(a)
    else:
        retty = a.dtype
        is_nan = get_isnan(retty)
        one = retty(1)

        def nancumprod_impl(a):
            out = np.empty(a.size, retty)
            c = one
            for idx, v in enumerate(a.flat):
                if ~is_nan(v):
                    c *= v
                out[idx] = c
            return out
        return nancumprod_impl