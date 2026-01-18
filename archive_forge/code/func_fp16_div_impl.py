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
@lower(operator.truediv, types.float16, types.float16)
@lower(operator.itruediv, types.float16, types.float16)
def fp16_div_impl(context, builder, sig, args):

    def fp16_div(x, y):
        return cuda.fp16.hdiv(x, y)
    return context.compile_internal(builder, fp16_div, sig, args)