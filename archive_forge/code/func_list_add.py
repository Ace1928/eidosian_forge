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
@lower_builtin(operator.add, types.List, types.List)
def list_add(context, builder, sig, args):
    a = ListInstance(context, builder, sig.args[0], args[0])
    b = ListInstance(context, builder, sig.args[1], args[1])
    a_size = a.size
    b_size = b.size
    nitems = builder.add(a_size, b_size)
    dest = ListInstance.allocate(context, builder, sig.return_type, nitems)
    dest.size = nitems
    with cgutils.for_range(builder, a_size) as loop:
        value = a.getitem(loop.index)
        value = context.cast(builder, value, a.dtype, dest.dtype)
        dest.setitem(loop.index, value, incref=True)
    with cgutils.for_range(builder, b_size) as loop:
        value = b.getitem(loop.index)
        value = context.cast(builder, value, b.dtype, dest.dtype)
        dest.setitem(builder.add(loop.index, a_size), value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, dest.value)