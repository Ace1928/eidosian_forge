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
@lower_builtin(operator.mul, types.List, types.Integer)
@lower_builtin(operator.mul, types.Integer, types.List)
def list_mul(context, builder, sig, args):
    if isinstance(sig.args[0], types.List):
        list_idx, int_idx = (0, 1)
    else:
        list_idx, int_idx = (1, 0)
    src = ListInstance(context, builder, sig.args[list_idx], args[list_idx])
    src_size = src.size
    mult = args[int_idx]
    zero = ir.Constant(mult.type, 0)
    mult = builder.select(cgutils.is_neg_int(builder, mult), zero, mult)
    nitems = builder.mul(mult, src_size)
    dest = ListInstance.allocate(context, builder, sig.return_type, nitems)
    dest.size = nitems
    with cgutils.for_range_slice(builder, zero, nitems, src_size, inc=True) as (dest_offset, _):
        with cgutils.for_range(builder, src_size) as loop:
            value = src.getitem(loop.index)
            dest.setitem(builder.add(loop.index, dest_offset), value, incref=True)
    return impl_ret_new_ref(context, builder, sig.return_type, dest.value)