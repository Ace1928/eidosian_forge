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
@overload(np.triu_indices)
def np_triu_indices(n, k=0, m=None):
    check_is_integer(n, 'n')
    check_is_integer(k, 'k')
    if not is_nonelike(m):
        check_is_integer(m, 'm')

    def np_triu_indices_impl(n, k=0, m=None):
        return np.nonzero(1 - np.tri(n, m, k=k - 1))
    return np_triu_indices_impl