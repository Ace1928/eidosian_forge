import itertools
from llvmlite import ir
from numba.core import cgutils, targetconfig
from .cudadrv import nvvm
def declare_string(builder, value):
    lmod = builder.basic_block.function.module
    cval = cgutils.make_bytearray(value.encode('utf-8') + b'\x00')
    gl = cgutils.add_global_variable(lmod, cval.type, name='_str', addrspace=nvvm.ADDRSPACE_CONSTANT)
    gl.linkage = 'internal'
    gl.global_constant = True
    gl.initializer = cval
    return builder.addrspacecast(gl, ir.PointerType(ir.IntType(8)), 'generic')