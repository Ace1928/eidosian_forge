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
def intersection_impl(a, b):
    if len(a) < len(b):
        s = a.copy()
        s.intersection_update(b)
        return s
    else:
        s = b.copy()
        s.intersection_update(a)
        return s