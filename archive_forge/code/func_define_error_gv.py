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
def define_error_gv(postfix):
    name = wrapfn.name + postfix
    gv = cgutils.add_global_variable(wrapper_module, ir.IntType(32), name)
    gv.initializer = ir.Constant(gv.type.pointee, None)
    return gv