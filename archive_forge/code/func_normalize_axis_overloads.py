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
@overload(normalize_axis)
def normalize_axis_overloads(func_name, arg_name, ndim, axis):
    if not isinstance(func_name, StringLiteral):
        raise errors.TypingError('func_name must be a str literal.')
    if not isinstance(arg_name, StringLiteral):
        raise errors.TypingError('arg_name must be a str literal.')
    msg = f'{func_name.literal_value}: Argument {arg_name.literal_value} out of bounds for dimensions of the array'

    def impl(func_name, arg_name, ndim, axis):
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(msg)
        return axis
    return impl