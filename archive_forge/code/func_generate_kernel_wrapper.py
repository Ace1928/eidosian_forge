import re
from functools import cached_property
import llvmlite.binding as ll
from llvmlite import ir
from numba.core import (cgutils, config, debuginfo, itanium_mangler, types,
from numba.core.dispatcher import Dispatcher
from numba.core.base import BaseContext
from numba.core.callconv import BaseCallConv, MinimalCallConv
from numba.core.typing import cmathdecl
from numba.core import datamodel
from .cudadrv import nvvm
from numba.cuda import codegen, nvvmutils, ufuncs
from numba.cuda.models import cuda_data_manager
def generate_kernel_wrapper(self, library, fndesc, kernel_name, debug, lineinfo, filename, linenum):
    """
        Generate the kernel wrapper in the given ``library``.
        The function being wrapped is described by ``fndesc``.
        The wrapper function is returned.
        """
    argtypes = fndesc.argtypes
    arginfo = self.get_arg_packer(argtypes)
    argtys = list(arginfo.argument_types)
    wrapfnty = ir.FunctionType(ir.VoidType(), argtys)
    wrapper_module = self.create_module('cuda.kernel.wrapper')
    fnty = ir.FunctionType(ir.IntType(32), [self.call_conv.get_return_type(types.pyobject)] + argtys)
    func = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)
    prefixed = itanium_mangler.prepend_namespace(func.name, ns='cudapy')
    wrapfn = ir.Function(wrapper_module, wrapfnty, prefixed)
    builder = ir.IRBuilder(wrapfn.append_basic_block(''))
    if debug or lineinfo:
        directives_only = lineinfo and (not debug)
        debuginfo = self.DIBuilder(module=wrapper_module, filepath=filename, cgctx=self, directives_only=directives_only)
        debuginfo.mark_subprogram(wrapfn, kernel_name, fndesc.args, argtypes, linenum)
        debuginfo.mark_location(builder, linenum)

    def define_error_gv(postfix):
        name = wrapfn.name + postfix
        gv = cgutils.add_global_variable(wrapper_module, ir.IntType(32), name)
        gv.initializer = ir.Constant(gv.type.pointee, None)
        return gv
    gv_exc = define_error_gv('__errcode__')
    gv_tid = []
    gv_ctaid = []
    for i in 'xyz':
        gv_tid.append(define_error_gv('__tid%s__' % i))
        gv_ctaid.append(define_error_gv('__ctaid%s__' % i))
    callargs = arginfo.from_arguments(builder, wrapfn.args)
    status, _ = self.call_conv.call_function(builder, func, types.void, argtypes, callargs)
    if debug:
        with cgutils.if_likely(builder, status.is_ok):
            builder.ret_void()
        with builder.if_then(builder.not_(status.is_python_exc)):
            old = ir.Constant(gv_exc.type.pointee, None)
            xchg = builder.cmpxchg(gv_exc, old, status.code, 'monotonic', 'monotonic')
            changed = builder.extract_value(xchg, 1)
            sreg = nvvmutils.SRegBuilder(builder)
            with builder.if_then(changed):
                for dim, ptr in zip('xyz', gv_tid):
                    val = sreg.tid(dim)
                    builder.store(val, ptr)
                for dim, ptr in zip('xyz', gv_ctaid):
                    val = sreg.ctaid(dim)
                    builder.store(val, ptr)
    builder.ret_void()
    nvvm.set_cuda_kernel(wrapfn)
    library.add_ir_module(wrapper_module)
    if debug or lineinfo:
        debuginfo.finalize()
    library.finalize()
    if config.DUMP_LLVM:
        utils.dump_llvm(fndesc, wrapper_module)
    return library.get_function(wrapfn.name)