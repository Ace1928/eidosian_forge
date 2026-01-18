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
def check_all_set(*args):
    if not all([isinstance(typ, types.Set) for typ in args]):
        raise TypingError(f'All arguments must be Sets, got {args}')
    if not all([args[0].dtype == s.dtype for s in args]):
        raise TypingError(f'All Sets must be of the same type, got {args}')