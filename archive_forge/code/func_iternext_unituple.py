import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin('iternext', types.UniTupleIter)
@iternext_impl(RefType.BORROWED)
def iternext_unituple(context, builder, sig, args, result):
    [tupiterty] = sig.args
    [tupiter] = args
    iterval = context.make_helper(builder, tupiterty, value=tupiter)
    tup = iterval.tuple
    idxptr = iterval.index
    idx = builder.load(idxptr)
    count = context.get_constant(types.intp, tupiterty.container.count)
    is_valid = builder.icmp_signed('<', idx, count)
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        getitem_sig = typing.signature(tupiterty.container.dtype, tupiterty.container, types.intp)
        getitem_out = getitem_unituple(context, builder, getitem_sig, [tup, idx])
        if context.enable_nrt:
            context.nrt.decref(builder, tupiterty.container.dtype, getitem_out)
        result.yield_(getitem_out)
        nidx = builder.add(idx, context.get_constant(types.intp, 1))
        builder.store(nidx, iterval.index)