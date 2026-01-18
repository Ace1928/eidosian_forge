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
def set_intersection_update(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    other = SetInstance(context, builder, sig.args[1], args[1])
    inst.intersect(other)
    return context.get_dummy_value()