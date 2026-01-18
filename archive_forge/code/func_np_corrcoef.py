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
@overload(np.corrcoef)
def np_corrcoef(x, y=None, rowvar=True):
    x_dt = determine_dtype(x)
    y_dt = determine_dtype(y)
    dtype = np.result_type(x_dt, y_dt, np.float64)
    if dtype == np.complex_:
        clip_fn = _clip_complex
    else:
        clip_fn = _clip_corr

    def np_corrcoef_impl(x, y=None, rowvar=True):
        c = np.cov(x, y, rowvar)
        d = np.diag(c)
        stddev = np.sqrt(d.real)
        for i in range(c.shape[0]):
            c[i, :] /= stddev
            c[:, i] /= stddev
        return clip_fn(c)

    def np_corrcoef_impl_single_variable(x, y=None, rowvar=True):
        c = np.cov(x, y, rowvar)
        return c / c
    if scalar_result_expected(x, y):
        return np_corrcoef_impl_single_variable
    else:
        return np_corrcoef_impl