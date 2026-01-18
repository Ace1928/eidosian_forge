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
def normalize_reshape_value(origsize, shape):
    num_neg_value = 0
    known_size = 1
    for ax, s in enumerate(shape):
        if s < 0:
            num_neg_value += 1
            neg_ax = ax
        else:
            known_size *= s
    if num_neg_value == 0:
        if origsize != known_size:
            raise ValueError('total size of new array must be unchanged')
    elif num_neg_value == 1:
        if known_size == 0:
            inferred = 0
            ok = origsize == 0
        else:
            inferred = origsize // known_size
            ok = origsize % known_size == 0
        if not ok:
            raise ValueError('total size of new array must be unchanged')
        shape[neg_ax] = inferred
    else:
        raise ValueError('multiple negative shape values')