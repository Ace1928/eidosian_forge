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
@overload(np.convolve)
def np_convolve(a, v, mode='full'):
    _assert_1d(a, 'np.convolve')
    _assert_1d(v, 'np.convolve')

    def impl(a, v, mode='full'):
        la = len(a)
        lv = len(v)
        if la == 0:
            raise ValueError("'a' cannot be empty")
        if lv == 0:
            raise ValueError("'v' cannot be empty")
        if la < lv:
            return _np_correlate_core(v, a[::-1], mode, 1)
        else:
            return _np_correlate_core(a, v[::-1], mode, 1)
    return impl