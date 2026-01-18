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
def real_sign_impl(context, builder, sig, args):
    """
    np.sign(float)
    """
    [x] = args
    POS = Constant(x.type, 1)
    NEG = Constant(x.type, -1)
    ZERO = Constant(x.type, 0)
    presult = cgutils.alloca_once(builder, x.type)
    is_pos = builder.fcmp_ordered('>', x, ZERO)
    is_neg = builder.fcmp_ordered('<', x, ZERO)
    with builder.if_else(is_pos) as (gt_zero, not_gt_zero):
        with gt_zero:
            builder.store(POS, presult)
        with not_gt_zero:
            with builder.if_else(is_neg) as (lt_zero, not_lt_zero):
                with lt_zero:
                    builder.store(NEG, presult)
                with not_lt_zero:
                    builder.store(x, presult)
    res = builder.load(presult)
    return impl_ret_untracked(context, builder, sig.return_type, res)