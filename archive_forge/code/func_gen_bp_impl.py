import os
import sys
from llvmlite import ir
from numba.core import types, utils, config, cgutils, errors
from numba import gdb, gdb_init, gdb_breakpoint
from numba.core.extending import overload, intrinsic
def gen_bp_impl():

    @intrinsic
    def bp_internal(tyctx):
        function_sig = types.void()

        def codegen(cgctx, builder, signature, args):
            mod = builder.module
            fnty = ir.FunctionType(ir.VoidType(), tuple())
            breakpoint = cgutils.get_or_insert_function(mod, fnty, 'numba_gdb_breakpoint')
            builder.call(breakpoint, tuple())
            return cgctx.get_constant(types.none, None)
        return (function_sig, codegen)
    return bp_internal