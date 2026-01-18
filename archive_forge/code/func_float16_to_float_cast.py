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
@lower_cast(types.float16, types.Float)
def float16_to_float_cast(context, builder, fromty, toty, val):
    if fromty.bitwidth == toty.bitwidth:
        return val
    ty, constraint = float16_float_ty_constraint(toty.bitwidth)
    fnty = ir.FunctionType(context.get_value_type(toty), [ir.IntType(16)])
    asm = ir.InlineAsm(fnty, f'cvt.{ty}.f16 $0, $1;', f'={constraint},h')
    return builder.call(asm, [val])