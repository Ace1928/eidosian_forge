from itertools import zip_longest
from llvmlite import ir
from numba.core import cgutils, types, typing, utils
from numba.core.imputils import (impl_ret_borrowed, impl_ret_new_ref,
@lower_builtin(slice, types.VarArg(types.Any))
def slice_constructor_impl(context, builder, sig, args):
    default_start_pos, default_start_neg, default_stop_pos, default_stop_neg, default_step = [context.get_constant(types.intp, x) for x in get_defaults(context)]
    slice_args = [None] * 3
    if len(args) == 1 and sig.args[0] is not types.none:
        slice_args[1] = args[0]
    else:
        for i, (ty, val) in enumerate(zip(sig.args, args)):
            if ty is not types.none:
                slice_args[i] = val

    def get_arg_value(i, default):
        val = slice_args[i]
        if val is None:
            return default
        else:
            return val
    step = get_arg_value(2, default_step)
    is_step_negative = builder.icmp_signed('<', step, context.get_constant(types.intp, 0))
    default_stop = builder.select(is_step_negative, default_stop_neg, default_stop_pos)
    default_start = builder.select(is_step_negative, default_start_neg, default_start_pos)
    stop = get_arg_value(1, default_stop)
    start = get_arg_value(0, default_start)
    ty = sig.return_type
    sli = context.make_helper(builder, sig.return_type)
    sli.start = start
    sli.stop = stop
    sli.step = step
    res = sli._getvalue()
    return impl_ret_untracked(context, builder, sig.return_type, res)