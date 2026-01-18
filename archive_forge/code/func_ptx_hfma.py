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
@lower(stubs.fp16.hfma, types.float16, types.float16, types.float16)
def ptx_hfma(context, builder, sig, args):
    argtys = [ir.IntType(16), ir.IntType(16), ir.IntType(16)]
    fnty = ir.FunctionType(ir.IntType(16), argtys)
    asm = ir.InlineAsm(fnty, 'fma.rn.f16 $0,$1,$2,$3;', '=h,h,h,h')
    return builder.call(asm, args)