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
@overload(np.intersect1d)
def jit_np_intersect1d(ar1, ar2):
    if not (type_can_asarray(ar1) or type_can_asarray(ar2)):
        raise TypingError('intersect1d: first two args must be array-like')

    def np_intersects1d_impl(ar1, ar2):
        ar1 = np.asarray(ar1)
        ar2 = np.asarray(ar2)
        ar1 = np.unique(ar1)
        ar2 = np.unique(ar2)
        aux = np.concatenate((ar1, ar2))
        aux.sort()
        mask = aux[1:] == aux[:-1]
        int1d = aux[:-1][mask]
        return int1d
    return np_intersects1d_impl