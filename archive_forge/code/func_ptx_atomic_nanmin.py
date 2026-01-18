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
@lower(stubs.atomic.nanmin, types.Array, types.intp, types.Any)
@lower(stubs.atomic.nanmin, types.Array, types.Tuple, types.Any)
@lower(stubs.atomic.nanmin, types.Array, types.UniTuple, types.Any)
@_atomic_dispatcher
def ptx_atomic_nanmin(context, builder, dtype, ptr, val):
    lmod = builder.module
    if dtype == types.float64:
        return builder.call(nvvmutils.declare_atomic_nanmin_float64(lmod), (ptr, val))
    elif dtype == types.float32:
        return builder.call(nvvmutils.declare_atomic_nanmin_float32(lmod), (ptr, val))
    elif dtype in (types.int32, types.int64):
        return builder.atomic_rmw('min', ptr, val, ordering='monotonic')
    elif dtype in (types.uint32, types.uint64):
        return builder.atomic_rmw('umin', ptr, val, ordering='monotonic')
    else:
        raise TypeError('Unimplemented atomic min with %s array' % dtype)