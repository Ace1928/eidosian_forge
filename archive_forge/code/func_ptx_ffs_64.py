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
@lower(stubs.ffs, types.i8)
@lower(stubs.ffs, types.u8)
def ptx_ffs_64(context, builder, sig, args):
    fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(ir.IntType(32), (ir.IntType(64),)), '__nv_ffsll')
    return builder.call(fn, args)