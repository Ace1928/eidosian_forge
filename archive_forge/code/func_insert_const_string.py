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
def insert_const_string(self, mod, string):
    """
        Unlike the parent version.  This returns a a pointer in the constant
        addrspace.
        """
    text = cgutils.make_bytearray(string.encode('utf-8') + b'\x00')
    name = '$'.join(['__conststring__', itanium_mangler.mangle_identifier(string)])
    gv = mod.globals.get(name)
    if gv is None:
        gv = cgutils.add_global_variable(mod, text.type, name, addrspace=nvvm.ADDRSPACE_CONSTANT)
        gv.linkage = 'internal'
        gv.global_constant = True
        gv.initializer = text
    charty = gv.type.pointee.element
    return gv.bitcast(charty.as_pointer(nvvm.ADDRSPACE_CONSTANT))