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
def int_sign_impl(context, builder, sig, args):
    """
    np.sign(int)
    """
    [x] = args
    POS = Constant(x.type, 1)
    NEG = Constant(x.type, -1)
    ZERO = Constant(x.type, 0)
    cmp_zero = builder.icmp_unsigned('==', x, ZERO)
    cmp_pos = builder.icmp_signed('>', x, ZERO)
    presult = cgutils.alloca_once(builder, x.type)
    bb_zero = builder.append_basic_block('.zero')
    bb_postest = builder.append_basic_block('.postest')
    bb_pos = builder.append_basic_block('.pos')
    bb_neg = builder.append_basic_block('.neg')
    bb_exit = builder.append_basic_block('.exit')
    builder.cbranch(cmp_zero, bb_zero, bb_postest)
    with builder.goto_block(bb_zero):
        builder.store(ZERO, presult)
        builder.branch(bb_exit)
    with builder.goto_block(bb_postest):
        builder.cbranch(cmp_pos, bb_pos, bb_neg)
    with builder.goto_block(bb_pos):
        builder.store(POS, presult)
        builder.branch(bb_exit)
    with builder.goto_block(bb_neg):
        builder.store(NEG, presult)
        builder.branch(bb_exit)
    builder.position_at_end(bb_exit)
    res = builder.load(presult)
    return impl_ret_untracked(context, builder, sig.return_type, res)