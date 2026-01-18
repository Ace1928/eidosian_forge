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
@lower_builtin('set.update', types.Set, types.IterableType)
def set_update(context, builder, sig, args):
    inst = SetInstance(context, builder, sig.args[0], args[0])
    items_type = sig.args[1]
    items = args[1]
    n = call_len(context, builder, items_type, items)
    if n is not None:
        new_size = builder.add(inst.payload.used, n)
        inst.upsize(new_size)
    with for_iter(context, builder, items_type, items) as loop:
        casted = context.cast(builder, loop.value, items_type.dtype, inst.dtype)
        inst.add(casted)
        context.nrt.decref(builder, items_type.dtype, loop.value)
    if n is not None:
        inst.downsize(inst.payload.used)
    return context.get_dummy_value()