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
@lower(round, types.f4, types.Integer)
@lower(round, types.f8, types.Integer)
def round_to_impl(context, builder, sig, args):

    def round_ndigits(x, ndigits):
        if math.isinf(x) or math.isnan(x):
            return x
        if ndigits >= 0:
            if ndigits > 22:
                pow1 = 10.0 ** (ndigits - 22)
                pow2 = 1e+22
            else:
                pow1 = 10.0 ** ndigits
                pow2 = 1.0
            y = x * pow1 * pow2
            if math.isinf(y):
                return x
        else:
            pow1 = 10.0 ** (-ndigits)
            y = x / pow1
        z = round(y)
        if math.fabs(y - z) == 0.5:
            z = 2.0 * round(y / 2.0)
        if ndigits >= 0:
            z = z / pow2 / pow1
        else:
            z *= pow1
        return z
    return context.compile_internal(builder, round_ndigits, sig, args)