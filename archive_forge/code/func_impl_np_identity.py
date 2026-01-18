import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@overload(np.identity)
def impl_np_identity(n, dtype=None):
    _check_const_str_dtype('identity', dtype)
    if not is_nonelike(dtype):
        nb_dtype = ty_parse_dtype(dtype)
    else:
        nb_dtype = types.double

    def identity(n, dtype=None):
        arr = np.zeros((n, n), nb_dtype)
        for i in range(n):
            arr[i, i] = 1
        return arr
    return identity