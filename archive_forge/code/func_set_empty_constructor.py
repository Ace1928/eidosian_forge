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
@lower_builtin(set)
def set_empty_constructor(context, builder, sig, args):
    set_type = sig.return_type
    inst = SetInstance.allocate(context, builder, set_type)
    return impl_ret_new_ref(context, builder, set_type, inst.value)