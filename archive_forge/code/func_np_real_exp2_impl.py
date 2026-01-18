import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_real_exp2_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 1)
    ll_ty = args[0].type
    fnty = llvmlite.ir.FunctionType(ll_ty, [ll_ty])
    fn = cgutils.insert_pure_function(builder.module, fnty, name='llvm.exp2')
    return builder.call(fn, [args[0]])