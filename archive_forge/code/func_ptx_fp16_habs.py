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
@lower(stubs.fp16.habs, types.float16)
def ptx_fp16_habs(context, builder, sig, args):
    fnty = ir.FunctionType(ir.IntType(16), [ir.IntType(16)])
    asm = ir.InlineAsm(fnty, 'abs.f16 $0, $1;', '=h,h')
    return builder.call(asm, args)