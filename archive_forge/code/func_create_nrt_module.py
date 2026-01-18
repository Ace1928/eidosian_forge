from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def create_nrt_module(ctx):
    """
    Create an IR module defining the LLVM NRT functions.
    A (IR module, library) tuple is returned.
    """
    codegen = ctx.codegen()
    library = codegen.create_library('nrt')
    ir_mod = library.create_ir_module('nrt_module')
    atomic_inc = _define_atomic_inc_dec(ir_mod, 'add', ordering='monotonic')
    atomic_dec = _define_atomic_inc_dec(ir_mod, 'sub', ordering='monotonic')
    _define_atomic_cas(ir_mod, ordering='monotonic')
    _define_nrt_meminfo_data(ir_mod)
    _define_nrt_incref(ir_mod, atomic_inc)
    _define_nrt_decref(ir_mod, atomic_dec)
    _define_nrt_unresolved_abort(ctx, ir_mod)
    return (ir_mod, library)