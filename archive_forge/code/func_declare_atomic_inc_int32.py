import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_inc_int32(lmod):
    fname = 'llvm.nvvm.atomic.load.inc.32.p0i32'
    fnty = ir.FunctionType(ir.IntType(32), (ir.PointerType(ir.IntType(32)), ir.IntType(32)))
    return cgutils.get_or_insert_function(lmod, fnty, fname)