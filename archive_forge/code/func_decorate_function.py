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
def decorate_function(self, fn, args, fe_argtypes, noalias=False):
    """
        Set names and attributes of function arguments.
        """
    assert not noalias
    arginfo = self._get_arg_packer(fe_argtypes)
    arginfo.assign_names(self.get_arguments(fn), ['arg.' + a for a in args])