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
@lower(stubs.cbrt, types.float32)
@lower(stubs.cbrt, types.float64)
def ptx_cbrt(context, builder, sig, args):
    ty = sig.return_type
    fname = cbrt_funcs[ty]
    fty = context.get_value_type(ty)
    lmod = builder.module
    fnty = ir.FunctionType(fty, [fty])
    fn = cgutils.get_or_insert_function(lmod, fnty, fname)
    return builder.call(fn, args)