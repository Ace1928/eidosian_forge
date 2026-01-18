import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_cbrt_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)

    @register_jitable(fastmath=True)
    def cbrt(x):
        if x < 0:
            return -np.power(-x, 1.0 / 3.0)
        else:
            return np.power(x, 1.0 / 3.0)

    def _cbrt(x):
        if np.isnan(x):
            return np.nan
        return cbrt(x)
    return context.compile_internal(builder, _cbrt, sig, args)