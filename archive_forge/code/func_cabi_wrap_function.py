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
def cabi_wrap_function(context, lib, fndesc, wrapper_function_name, nvvm_options):
    """
    Wrap a Numba ABI function in a C ABI wrapper at the NVVM IR level.

    The C ABI wrapper will have the same name as the source Python function.
    """
    library = lib.codegen.create_library(f'{lib.name}_function_', entry_name=wrapper_function_name, nvvm_options=nvvm_options)
    library.add_linking_library(lib)
    argtypes = fndesc.argtypes
    restype = fndesc.restype
    c_call_conv = CUDACABICallConv(context)
    wrapfnty = c_call_conv.get_function_type(restype, argtypes)
    fnty = context.call_conv.get_function_type(fndesc.restype, argtypes)
    wrapper_module = context.create_module('cuda.cabi.wrapper')
    func = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)
    wrapfn = ir.Function(wrapper_module, wrapfnty, wrapper_function_name)
    builder = ir.IRBuilder(wrapfn.append_basic_block(''))
    arginfo = context.get_arg_packer(argtypes)
    callargs = arginfo.from_arguments(builder, wrapfn.args)
    _, return_value = context.call_conv.call_function(builder, func, restype, argtypes, callargs)
    builder.ret(return_value)
    library.add_ir_module(wrapper_module)
    library.finalize()
    return library