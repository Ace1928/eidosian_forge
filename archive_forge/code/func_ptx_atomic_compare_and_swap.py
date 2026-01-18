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
@lower(stubs.atomic.compare_and_swap, types.Array, types.Any, types.Any)
def ptx_atomic_compare_and_swap(context, builder, sig, args):
    sig = sig.return_type(sig.args[0], types.intp, sig.args[1], sig.args[2])
    args = (args[0], context.get_constant(types.intp, 0), args[1], args[2])
    return ptx_atomic_cas(context, builder, sig, args)