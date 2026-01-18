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
@lower_builtin(divmod, types.Float, types.Float)
def real_divmod_impl(context, builder, sig, args, loc=None):
    x, y = args
    quot = cgutils.alloca_once(builder, x.type, name='quot')
    rem = cgutils.alloca_once(builder, x.type, name='rem')
    with builder.if_else(cgutils.is_scalar_zero(builder, y), likely=False) as (if_zero, if_non_zero):
        with if_zero:
            if not context.error_model.fp_zero_division(builder, ('modulo by zero',), loc):
                q = builder.fdiv(x, y)
                r = builder.frem(x, y)
                builder.store(q, quot)
                builder.store(r, rem)
        with if_non_zero:
            q, r = real_divmod(context, builder, x, y)
            builder.store(q, quot)
            builder.store(r, rem)
    return cgutils.pack_array(builder, (builder.load(quot), builder.load(rem)))