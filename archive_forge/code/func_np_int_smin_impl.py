import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_int_smin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_sle_arg2 = builder.icmp_signed('<=', arg1, arg2)
    return builder.select(arg1_sle_arg2, arg1, arg2)