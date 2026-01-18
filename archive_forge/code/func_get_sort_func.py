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
def get_sort_func(kind, lt_impl, is_argsort=False):
    """
    Get a sort implementation of the given kind.
    """
    key = (kind, lt_impl.__name__, is_argsort)
    try:
        return _sorts[key]
    except KeyError:
        if kind == 'quicksort':
            sort = quicksort.make_jit_quicksort(lt=lt_impl, is_argsort=is_argsort, is_np_array=True)
            func = sort.run_quicksort
        elif kind == 'mergesort':
            sort = mergesort.make_jit_mergesort(lt=lt_impl, is_argsort=is_argsort)
            func = sort.run_mergesort
        _sorts[key] = func
        return func