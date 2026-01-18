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
def digitize_impl(x, bins, right=False):
    mono = _monotonicity(bins)
    if mono == 0:
        raise ValueError('bins must be monotonically increasing or decreasing')
    if right:
        if mono == -1:
            return len(bins) - np.searchsorted(bins[::-1], x, side='left')
        else:
            return np.searchsorted(bins, x, side='left')
    elif mono == -1:
        return len(bins) - np.searchsorted(bins[::-1], x, side='right')
    else:
        return np.searchsorted(bins, x, side='right')