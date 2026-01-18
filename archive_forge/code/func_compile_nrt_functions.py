from numba.core import config
from numba.core import types, cgutils
from llvmlite import ir, binding
def compile_nrt_functions(ctx):
    """
    Compile all LLVM NRT functions and return a library containing them.
    The library is created using the given target context.
    """
    ir_mod, library = create_nrt_module(ctx)
    library.add_ir_module(ir_mod)
    library.finalize()
    return library