import operator
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import types
from numba.core.extending import overload_method
@lower_builtin(operator.eq, types.EnumMember, types.EnumMember)
def enum_eq(context, builder, sig, args):
    tu, tv = sig.args
    u, v = args
    res = context.generic_compare(builder, operator.eq, (tu.dtype, tv.dtype), (u, v))
    return impl_ret_untracked(context, builder, sig.return_type, res)