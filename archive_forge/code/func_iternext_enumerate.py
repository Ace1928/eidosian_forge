from numba.core import types, cgutils
from numba.core.imputils import (
@lower_builtin('iternext', types.EnumerateType)
@iternext_impl(RefType.NEW)
def iternext_enumerate(context, builder, sig, args, result):
    [enumty] = sig.args
    [enum] = args
    enum = context.make_helper(builder, enumty, value=enum)
    count = builder.load(enum.count)
    ncount = builder.add(count, context.get_constant(types.intp, 1))
    builder.store(ncount, enum.count)
    srcres = call_iternext(context, builder, enumty.source_type, enum.iter)
    is_valid = srcres.is_valid()
    result.set_valid(is_valid)
    with builder.if_then(is_valid):
        srcval = srcres.yielded_value()
        result.yield_(context.make_tuple(builder, enumty.yield_type, [count, srcval]))