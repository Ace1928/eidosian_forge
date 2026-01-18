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
def insert_string_const_addrspace(self, builder, string):
    """
        Insert a constant string in the constant addresspace and return a
        generic i8 pointer to the data.

        This function attempts to deduplicate.
        """
    lmod = builder.module
    gv = self.insert_const_string(lmod, string)
    charptrty = ir.PointerType(ir.IntType(8))
    return builder.addrspacecast(gv, charptrty, 'generic')