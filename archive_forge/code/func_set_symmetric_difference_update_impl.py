import collections
import contextlib
import math
import operator
from functools import cached_property
from llvmlite import ir
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.misc import quicksort
from numba.cpython import slicing
from numba.core.errors import NumbaValueError, TypingError
from numba.core.extending import overload, overload_method, intrinsic
@overload_method(types.Set, 'symmetric_difference_update')
def set_symmetric_difference_update_impl(a, b):
    check_all_set(a, b)
    return lambda a, b: _set_symmetric_difference_update(a, b)