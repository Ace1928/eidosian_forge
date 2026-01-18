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
@lower(stubs.match_any_sync, types.i4, types.i4)
@lower(stubs.match_any_sync, types.i4, types.i8)
@lower(stubs.match_any_sync, types.i4, types.f4)
@lower(stubs.match_any_sync, types.i4, types.f8)
def ptx_match_any_sync(context, builder, sig, args):
    mask, value = args
    width = sig.args[1].bitwidth
    if sig.args[1] in types.real_domain:
        value = builder.bitcast(value, ir.IntType(width))
    fname = 'llvm.nvvm.match.any.sync.i{}'.format(width)
    lmod = builder.module
    fnty = ir.FunctionType(ir.IntType(32), (ir.IntType(32), ir.IntType(width)))
    func = cgutils.get_or_insert_function(lmod, fnty, fname)
    return builder.call(func, (mask, value))