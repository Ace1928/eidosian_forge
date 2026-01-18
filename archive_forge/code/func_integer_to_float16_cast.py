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
@lower_cast(types.Integer, types.float16)
@lower_cast(types.IntegerLiteral, types.float16)
def integer_to_float16_cast(context, builder, fromty, toty, val):
    bitwidth = fromty.bitwidth
    constraint = float16_int_constraint(bitwidth)
    signedness = 's' if fromty.signed else 'u'
    fnty = ir.FunctionType(ir.IntType(16), [context.get_value_type(fromty)])
    asm = ir.InlineAsm(fnty, f'cvt.rn.f16.{signedness}{bitwidth} $0, $1;', f'=h,{constraint}')
    return builder.call(asm, [val])