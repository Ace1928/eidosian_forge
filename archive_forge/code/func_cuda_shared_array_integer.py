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
@lower(cuda.shared.array, types.IntegerLiteral, types.Any)
def cuda_shared_array_integer(context, builder, sig, args):
    length = sig.args[0].literal_value
    dtype = parse_dtype(sig.args[1])
    return _generic_array(context, builder, shape=(length,), dtype=dtype, symbol_name=_get_unique_smem_id('_cudapy_smem'), addrspace=nvvm.ADDRSPACE_SHARED, can_dynsized=True)