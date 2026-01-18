import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_nanmax_float64(lmod):
    fname = '___numba_atomic_double_nanmax'
    fnty = ir.FunctionType(ir.DoubleType(), (ir.PointerType(ir.DoubleType()), ir.DoubleType()))
    return cgutils.get_or_insert_function(lmod, fnty, fname)