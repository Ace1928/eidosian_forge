import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def llvm_to_ptx(llvmir, **opts):
    if isinstance(llvmir, str):
        llvmir = [llvmir]
    if opts.pop('fastmath', False):
        opts.update({'ftz': True, 'fma': True, 'prec_div': False, 'prec_sqrt': False})
    cu = CompilationUnit()
    libdevice = LibDevice()
    for mod in llvmir:
        mod = llvm_replace(mod)
        cu.add_module(mod.encode('utf8'))
    cu.lazy_add_module(libdevice.get())
    return cu.compile(**opts)