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
def get_function_type(self, restype, argtypes):
    """
        Get the LLVM IR Function type for *restype* and *argtypes*.
        """
    arginfo = self._get_arg_packer(argtypes)
    argtypes = list(arginfo.argument_types)
    fnty = ir.FunctionType(self.get_return_type(restype), argtypes)
    return fnty