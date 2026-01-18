import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, errors, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.extending import overload_method, overload
from numba.misc import quicksort
from numba.cpython import slicing
from numba import literal_unroll
@overload_method(types.List, 'reverse')
def list_reverse(lst):

    def list_reverse_impl(lst):
        for a in range(0, len(lst) // 2):
            b = -a - 1
            lst[a], lst[b] = (lst[b], lst[a])
    return list_reverse_impl