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
def real_mod_impl(context, builder, sig, args, loc=None):
    x, y = args
    res = cgutils.alloca_once(builder, x.type)
    with builder.if_else(cgutils.is_scalar_zero(builder, y), likely=False) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(builder, ('modulo by zero',), loc):
                rem = builder.frem(x, y)
                builder.store(rem, res)
        with if_non_zero:
            _, rem = real_divmod(context, builder, x, y)
            builder.store(rem, res)
    return impl_ret_untracked(context, builder, sig.return_type, builder.load(res))