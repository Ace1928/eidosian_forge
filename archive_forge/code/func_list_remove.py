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
@overload_method(types.List, 'remove')
def list_remove(lst, value):

    def list_remove_impl(lst, value):
        for i in range(len(lst)):
            if lst[i] == value:
                lst.pop(i)
                return
        raise ValueError('list.remove(x): x not in list')
    return list_remove_impl