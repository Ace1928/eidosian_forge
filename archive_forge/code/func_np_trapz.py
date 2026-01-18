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
@overload(np.trapz)
def np_trapz(y, x=None, dx=1.0):
    if isinstance(y, (types.Number, types.Boolean)):
        raise TypingError('y cannot be a scalar')
    elif isinstance(y, types.Array) and y.ndim == 0:
        raise TypingError('y cannot be 0D')

    def impl(y, x=None, dx=1.0):
        yarr = np.asarray(y)
        d = _get_d(x, dx)
        y_ave = (yarr[..., slice(1, None)] + yarr[..., slice(None, -1)]) / 2.0
        ret = np.sum(d * y_ave, -1)
        processed = _select_element(ret)
        return processed
    return impl