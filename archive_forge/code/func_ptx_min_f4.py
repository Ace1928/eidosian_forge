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
@lower(min, types.f4, types.f4)
def ptx_min_f4(context, builder, sig, args):
    fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.FloatType(), (ir.FloatType(), ir.FloatType())), '__nv_fminf')
    return builder.call(fn, args)