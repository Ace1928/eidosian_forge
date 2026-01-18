import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_nanmin_float32(lmod):
    fname = '___numba_atomic_float_nanmin'
    fnty = ir.FunctionType(ir.FloatType(), (ir.PointerType(ir.FloatType()), ir.FloatType()))
    return cgutils.get_or_insert_function(lmod, fnty, fname)