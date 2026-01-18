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
def prepare_cuda_kernel(self, codelib, fndesc, debug, lineinfo, nvvm_options, filename, linenum, max_registers=None):
    """
        Adapt a code library ``codelib`` with the numba compiled CUDA kernel
        with name ``fname`` and arguments ``argtypes`` for NVVM.
        A new library is created with a wrapper function that can be used as
        the kernel entry point for the given kernel.

        Returns the new code library and the wrapper function.

        Parameters:

        codelib:       The CodeLibrary containing the device function to wrap
                       in a kernel call.
        fndesc:        The FunctionDescriptor of the source function.
        debug:         Whether to compile with debug.
        lineinfo:      Whether to emit line info.
        nvvm_options:  Dict of NVVM options used when compiling the new library.
        filename:      The source filename that the function is contained in.
        linenum:       The source line that the function is on.
        max_registers: The max_registers argument for the code library.
        """
    kernel_name = itanium_mangler.prepend_namespace(fndesc.llvm_func_name, ns='cudapy')
    library = self.codegen().create_library(f'{codelib.name}_kernel_', entry_name=kernel_name, nvvm_options=nvvm_options, max_registers=max_registers)
    library.add_linking_library(codelib)
    wrapper = self.generate_kernel_wrapper(library, fndesc, kernel_name, debug, lineinfo, filename, linenum)
    return (library, wrapper)