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
@overload(np.nanvar)
def np_nanvar(a):
    if not isinstance(a, types.Array):
        return
    isnan = get_isnan(a.dtype)

    def nanvar_impl(a):
        m = np.nanmean(a)
        ssd = 0.0
        count = 0
        for view in np.nditer(a):
            v = view.item()
            if not isnan(v):
                val = v.item() - m
                ssd += np.real(val * np.conj(val))
                count += 1
        return np.divide(ssd, count)
    return nanvar_impl