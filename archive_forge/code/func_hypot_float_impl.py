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
@lower(math.hypot, types.Float, types.Float)
def hypot_float_impl(context, builder, sig, args):
    xty, yty = sig.args
    assert xty == yty == sig.return_type
    x, y = args
    fname = {types.float32: '_hypotf' if sys.platform == 'win32' else 'hypotf', types.float64: '_hypot' if sys.platform == 'win32' else 'hypot'}[xty]
    plat_hypot = types.ExternalFunction(fname, sig)
    if sys.platform == 'win32' and config.MACHINE_BITS == 32:
        inf = xty(float('inf'))

        def hypot_impl(x, y):
            if math.isinf(x) or math.isinf(y):
                return inf
            return plat_hypot(x, y)
    else:

        def hypot_impl(x, y):
            return plat_hypot(x, y)
    res = context.compile_internal(builder, hypot_impl, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)