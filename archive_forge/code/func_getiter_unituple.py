import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin('getiter', types.UniTuple)
@lower_builtin('getiter', types.NamedUniTuple)
def getiter_unituple(context, builder, sig, args):
    [tupty] = sig.args
    [tup] = args
    iterval = context.make_helper(builder, types.UniTupleIter(tupty))
    index0 = context.get_constant(types.intp, 0)
    indexptr = cgutils.alloca_once(builder, index0.type)
    builder.store(index0, indexptr)
    iterval.index = indexptr
    iterval.tuple = tup
    res = iterval._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)