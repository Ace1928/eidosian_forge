from functools import reduce
import operator
import math
from llvmlite import ir
import llvmlite.binding as ll
from numba.core.imputils import Registry, lower_cast
from numba.core.typing.npydecl import parse_dtype
from numba.core.datamodel import models
from numba.core import types, cgutils
from numba.np import ufunc_db
from numba.np.npyimpl import register_ufuncs
from .cudadrv import nvvm
from numba import cuda
from numba.cuda import nvvmutils, stubs, errors
from numba.cuda.types import dim3, CUDADispatcher
@lower_cast(types.Float, types.float16)
def float_to_float16_cast(context, builder, fromty, toty, val):
    if fromty.bitwidth == toty.bitwidth:
        return val
    ty, constraint = float16_float_ty_constraint(fromty.bitwidth)
    fnty = ir.FunctionType(ir.IntType(16), [context.get_value_type(fromty)])
    asm = ir.InlineAsm(fnty, f'cvt.rn.f16.{ty} $0, $1;', f'=h,{constraint}')
    return builder.call(asm, [val])