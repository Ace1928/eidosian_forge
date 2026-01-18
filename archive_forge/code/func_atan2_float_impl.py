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
@lower(math.atan2, types.Float, types.Float)
def atan2_float_impl(context, builder, sig, args):
    assert len(args) == 2
    mod = builder.module
    ty = sig.args[0]
    lty = context.get_value_type(ty)
    func_name = {types.float32: 'atan2f', types.float64: 'atan2'}[ty]
    fnty = llvmlite.ir.FunctionType(lty, (lty, lty))
    fn = cgutils.insert_pure_function(builder.module, fnty, name=func_name)
    res = builder.call(fn, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)