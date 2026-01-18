import operator
from numba.core import types, typing, cgutils
from numba.core.imputils import (lower_cast, lower_builtin,
def optional_is_none(context, builder, sig, args):
    """
    Check if an Optional value is invalid
    """
    [lty, rty] = sig.args
    [lval, rval] = args
    if lty == types.none:
        lty, rty = (rty, lty)
        lval, rval = (rval, lval)
    opt_type = lty
    opt_val = lval
    opt = context.make_helper(builder, opt_type, opt_val)
    res = builder.not_(cgutils.as_bool_bit(builder, opt.valid))
    return impl_ret_untracked(context, builder, sig.return_type, res)