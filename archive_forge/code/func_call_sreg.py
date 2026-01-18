import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def call_sreg(builder, name):
    module = builder.module
    fnty = ir.FunctionType(ir.IntType(32), ())
    fn = cgutils.get_or_insert_function(module, fnty, SREG_MAPPING[name])
    return builder.call(fn, ())