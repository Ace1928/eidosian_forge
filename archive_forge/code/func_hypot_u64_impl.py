import math
import operator
import sys
import numpy as np
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core.imputils import Registry, impl_ret_untracked
from numba import typeof
from numba.core import types, utils, config, cgutils
from numba.core.extending import overload
from numba.core.typing import signature
from numba.cpython.unsafe.numbers import trailing_zeros
@lower(math.hypot, types.uint64, types.uint64)
def hypot_u64_impl(context, builder, sig, args):
    [x, y] = args
    y = builder.sitofp(y, llvmlite.ir.DoubleType())
    x = builder.sitofp(x, llvmlite.ir.DoubleType())
    fsig = signature(types.float64, types.float64, types.float64)
    res = hypot_float_impl(context, builder, fsig, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)