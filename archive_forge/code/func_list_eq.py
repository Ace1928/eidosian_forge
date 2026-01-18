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
@lower_builtin(operator.eq, types.List, types.List)
def list_eq(context, builder, sig, args):
    aty, bty = sig.args
    a = ListInstance(context, builder, aty, args[0])
    b = ListInstance(context, builder, bty, args[1])
    a_size = a.size
    same_size = builder.icmp_signed('==', a_size, b.size)
    res = cgutils.alloca_once_value(builder, same_size)
    with builder.if_then(same_size):
        with cgutils.for_range(builder, a_size) as loop:
            v = a.getitem(loop.index)
            w = b.getitem(loop.index)
            itemres = context.generic_compare(builder, operator.eq, (aty.dtype, bty.dtype), (v, w))
            with builder.if_then(builder.not_(itemres)):
                builder.store(cgutils.false_bit, res)
                loop.do_break()
    return builder.load(res)