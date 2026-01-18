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
@overload(np.array)
def impl_np_array(object, dtype=None):
    _check_const_str_dtype('array', dtype)
    if not type_can_asarray(object):
        raise errors.TypingError('The argument "object" must be array-like')
    if not is_nonelike(dtype) and ty_parse_dtype(dtype) is None:
        msg = 'The argument "dtype" must be a data-type if it is provided'
        raise errors.TypingError(msg)

    def impl(object, dtype=None):
        return np_array(object, dtype)
    return impl