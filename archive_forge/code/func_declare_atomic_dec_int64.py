import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_dec_int64(lmod):
    fname = '___numba_atomic_u64_dec'
    fnty = ir.FunctionType(ir.IntType(64), (ir.PointerType(ir.IntType(64)), ir.IntType(64)))
    return cgutils.get_or_insert_function(lmod, fnty, fname)