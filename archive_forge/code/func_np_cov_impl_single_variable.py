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
def np_cov_impl_single_variable(m, y=None, rowvar=True, bias=False, ddof=None):
    X = _prepare_cov_input(m, y, rowvar, ddof, dtype, _DDOF_HANDLER, _M_DIM_HANDLER).astype(dtype)
    if np.any(np.array(X.shape) == 0):
        variance = np.nan
    else:
        variance = np_cov_impl_inner(X, bias, ddof).flat[0]
    return np.array(variance)