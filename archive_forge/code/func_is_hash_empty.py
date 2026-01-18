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
def is_hash_empty(context, builder, h):
    """
    Whether the hash value denotes an empty entry.
    """
    empty = ir.Constant(h.type, EMPTY)
    return builder.icmp_unsigned('==', h, empty)