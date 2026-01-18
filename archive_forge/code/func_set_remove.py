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
def set_remove(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    item = args[1]
    found = inst.discard(item)
    with builder.if_then(builder.not_(found), likely=False):
        context.call_conv.return_user_exc(builder, KeyError, ('set.remove(): key not in set',))
    return context.get_dummy_value()