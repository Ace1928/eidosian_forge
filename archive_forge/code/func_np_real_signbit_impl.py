import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_signbit_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1, return_type=types.boolean)
    masks = {types.float16: context.get_constant(types.uint16, 32768), types.float32: context.get_constant(types.uint32, 2147483648), types.float64: context.get_constant(types.uint64, 9223372036854775808)}
    arg_ty = sig.args[0]
    arg_int_ty = getattr(types, f'uint{arg_ty.bitwidth}')
    arg_ll_int_ty = context.get_value_type(arg_int_ty)
    int_res = builder.and_(builder.bitcast(args[0], arg_ll_int_ty), masks[arg_ty])
    bool_res = builder.icmp_unsigned('!=', int_res, int_res.type(0))
    return bool_res