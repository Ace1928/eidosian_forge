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
@overload(np.split)
def np_split(ary, indices_or_sections, axis=0):
    if isinstance(ary, (types.UniTuple, types.ListType, types.List)):

        def impl(ary, indices_or_sections, axis=0):
            return np.split(np.asarray(ary), indices_or_sections, axis=axis)
        return impl
    if isinstance(indices_or_sections, types.Integer):

        def impl(ary, indices_or_sections, axis=0):
            _, rem = divmod(ary.shape[axis], indices_or_sections)
            if rem != 0:
                raise ValueError('array split does not result in an equal division')
            return np.array_split(ary, indices_or_sections, axis=axis)
        return impl
    else:
        return np_array_split(ary, indices_or_sections, axis=axis)