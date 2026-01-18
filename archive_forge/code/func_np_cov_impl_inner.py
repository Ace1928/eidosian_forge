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
def np_cov_impl_inner(X, bias, ddof):
    if ddof is None:
        if bias:
            ddof = 0
        else:
            ddof = 1
    fact = X.shape[1] - ddof
    fact = max(fact, 0.0)
    X -= row_wise_average(X)
    c = np.dot(X, np.conj(X.T))
    c *= np.true_divide(1, fact)
    return c