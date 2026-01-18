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
@lower(stubs.syncwarp)
def ptx_syncwarp(context, builder, sig, args):
    mask = context.get_constant(types.int32, 4294967295)
    mask_sig = types.none(types.int32)
    return ptx_syncwarp_mask(context, builder, mask_sig, [mask])