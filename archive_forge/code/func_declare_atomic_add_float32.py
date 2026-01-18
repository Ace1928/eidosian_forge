import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_atomic_add_float32(lmod):
    fname = 'llvm.nvvm.atomic.load.add.f32.p0f32'
    fnty = ir.FunctionType(ir.FloatType(), (ir.PointerType(ir.FloatType(), 0), ir.FloatType()))
    return cgutils.get_or_insert_function(lmod, fnty, fname)