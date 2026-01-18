from llvmlite import ir
from numba.core.typing.templates import ConcreteTemplate
from numba.core import types, typing, funcdesc, config, compiler, sigutils
from numba.core.compiler import (sanitize_compile_result_entries, CompilerBase,
from numba.core.compiler_lock import global_compiler_lock
from numba.core.compiler_machinery import (LoweringPass,
from numba.core.errors import NumbaInvalidConfigWarning
from numba.core.typed_passes import (IRLegalization, NativeLowering,
from warnings import warn
from numba.cuda.api import get_current_device
from numba.cuda.target import CUDACABICallConv
@global_compiler_lock
def compile_cuda(pyfunc, return_type, args, debug=False, lineinfo=False, inline=False, fastmath=False, nvvm_options=None, cc=None):
    if cc is None:
        raise ValueError('Compute Capability must be supplied')
    from .descriptor import cuda_target
    typingctx = cuda_target.typing_context
    targetctx = cuda_target.target_context
    flags = CUDAFlags()
    flags.no_compile = True
    flags.no_cpython_wrapper = True
    flags.no_cfunc_wrapper = True
    if debug or lineinfo:
        flags.debuginfo = True
    if lineinfo:
        flags.dbg_directives_only = True
    if debug:
        flags.error_model = 'python'
    else:
        flags.error_model = 'numpy'
    if inline:
        flags.forceinline = True
    if fastmath:
        flags.fastmath = True
    if nvvm_options:
        flags.nvvm_options = nvvm_options
    flags.compute_capability = cc
    from numba.core.target_extension import target_override
    with target_override('cuda'):
        cres = compiler.compile_extra(typingctx=typingctx, targetctx=targetctx, func=pyfunc, args=args, return_type=return_type, flags=flags, locals={}, pipeline_class=CUDACompiler)
    library = cres.library
    library.finalize()
    return cres