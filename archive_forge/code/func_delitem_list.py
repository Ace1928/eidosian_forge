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
@lower_builtin(operator.delitem, types.List, types.SliceType)
def delitem_list(context, builder, sig, args):
    inst = ListInstance(context, builder, sig.args[0], args[0])
    slice = context.make_helper(builder, sig.args[1], args[1])
    slicing.guard_invalid_slice(context, builder, sig.args[1], slice)
    inst.fix_slice(slice)
    slice_len = slicing.get_slice_length(builder, slice)
    one = ir.Constant(slice_len.type, 1)
    with builder.if_then(builder.icmp_signed('!=', slice.step, one), likely=False):
        msg = 'unsupported del list[start:stop:step] with step != 1'
        context.call_conv.return_user_exc(builder, NotImplementedError, (msg,))
    start = slice.start
    real_stop = builder.add(start, slice_len)
    with cgutils.for_range_slice(builder, start, real_stop, start.type(1)) as (idx, _):
        inst.decref_value(inst.getitem(idx))
    tail_size = builder.sub(inst.size, real_stop)
    inst.move(start, real_stop, tail_size)
    inst.resize(builder.sub(inst.size, slice_len))
    return context.get_dummy_value()