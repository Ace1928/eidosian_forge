import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
def real_power_impl(context, builder, sig, args):
    x, y = args
    module = builder.module
    if context.implement_powi_as_math_call:
        imp = context.get_function(math.pow, sig)
        res = imp(builder, args)
    else:
        fn = module.declare_intrinsic('llvm.pow', [y.type])
        res = builder.call(fn, (x, y))
    return impl_ret_untracked(context, builder, sig.return_type, res)