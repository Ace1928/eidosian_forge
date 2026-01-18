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
@overload(np.logspace)
def numpy_logspace(start, stop, num=50):
    if not isinstance(start, types.Number):
        raise errors.TypingError('The first argument "start" must be a number')
    if not isinstance(stop, types.Number):
        raise errors.TypingError('The second argument "stop" must be a number')
    if not isinstance(num, (int, types.Integer)):
        raise errors.TypingError('The third argument "num" must be an integer')

    def impl(start, stop, num=50):
        y = np.linspace(start, stop, num)
        return np.power(10.0, y)
    return impl