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
@overload_method(types.List, 'sort')
def ol_list_sort(lst, key=None, reverse=False):
    _sort_check_key(key)
    _sort_check_reverse(reverse)
    if cgutils.is_nonelike(key):
        KEY = False
        sort_f = sort_forwards
        sort_b = sort_backwards
    elif isinstance(key, types.Dispatcher):
        KEY = True
        sort_f = arg_sort_forwards
        sort_b = arg_sort_backwards

    def impl(lst, key=None, reverse=False):
        if KEY is True:
            _lst = [key(x) for x in lst]
        else:
            _lst = lst
        if reverse is False or reverse == 0:
            tmp = sort_f(_lst)
        else:
            tmp = sort_b(_lst)
        if KEY is True:
            lst[:] = [lst[i] for i in tmp]
    return impl