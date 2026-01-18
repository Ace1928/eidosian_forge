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
@overload(sorted)
def ol_sorted(iterable, key=None, reverse=False):
    if not isinstance(iterable, types.IterableType):
        return False
    _sort_check_key(key)
    _sort_check_reverse(reverse)

    def impl(iterable, key=None, reverse=False):
        lst = list(iterable)
        lst.sort(key=key, reverse=reverse)
        return lst
    return impl