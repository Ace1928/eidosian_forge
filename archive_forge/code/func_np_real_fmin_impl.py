import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_fmin_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    arg1, arg2 = args
    arg1_nan = builder.fcmp_unordered('uno', arg1, arg1)
    any_nan = builder.fcmp_unordered('uno', arg1, arg2)
    nan_result = builder.select(arg1_nan, arg2, arg1)
    arg1_le_arg2 = builder.fcmp_ordered('<=', arg1, arg2)
    non_nan_result = builder.select(arg1_le_arg2, arg1, arg2)
    return builder.select(any_nan, nan_result, non_nan_result)