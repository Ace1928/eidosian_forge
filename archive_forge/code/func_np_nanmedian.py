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
@overload(np.nanmedian)
def np_nanmedian(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanmedian_impl(a):
        temp_arry = np.empty(a.size, a.dtype)
        n = 0
        for view in np.nditer(a):
            v = view.item()
            if not isnan(v):
                temp_arry[n] = v
                n += 1
        if n == 0:
            return np.nan
        return _median_inner(temp_arry, n)
    return nanmedian_impl