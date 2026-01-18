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
def ptx_atomic_bitwise(stub, op):

    @_atomic_dispatcher
    def impl_ptx_atomic(context, builder, dtype, ptr, val):
        if dtype in cuda.cudadecl.integer_numba_types:
            return builder.atomic_rmw(op, ptr, val, 'monotonic')
        else:
            raise TypeError(f'Unimplemented atomic {op} with {dtype} array')
    for ty in (types.intp, types.UniTuple, types.Tuple):
        lower(stub, types.Array, ty, types.Any)(impl_ptx_atomic)