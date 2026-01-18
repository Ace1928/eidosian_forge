import operator
from numba.core.imputils import (lower_builtin, lower_getattr_generic,
from numba.core import typing, types, cgutils
from numba.core.extending import overload_method, overload, intrinsic
@lower_builtin(types.NamedTupleClass, types.VarArg(types.Any))
def namedtuple_constructor(context, builder, sig, args):
    newargs = []
    for i, arg in enumerate(args):
        casted = context.cast(builder, arg, sig.args[i], sig.return_type[i])
        newargs.append(casted)
    res = context.make_tuple(builder, sig.return_type, tuple(newargs))
    return impl_ret_borrowed(context, builder, sig.return_type, res)