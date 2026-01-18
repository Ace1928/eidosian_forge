import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_cudaCGGetIntrinsicHandle(lmod):
    fname = 'cudaCGGetIntrinsicHandle'
    fnty = ir.FunctionType(ir.IntType(64), (ir.IntType(32),))
    return cgutils.get_or_insert_function(lmod, fnty, fname)