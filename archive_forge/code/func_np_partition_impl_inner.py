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
@register_jitable
def np_partition_impl_inner(a, kth_array):
    out = np.empty_like(a)
    idx = np.ndindex(a.shape[:-1])
    for s in idx:
        arry = a[s].copy()
        low = 0
        high = len(arry) - 1
        for kth in kth_array:
            _select_w_nan(arry, kth, low, high)
            low = kth
        out[s] = arry
    return out