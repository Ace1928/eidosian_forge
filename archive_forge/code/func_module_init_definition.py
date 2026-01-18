import logging
import os
import sys
from llvmlite import ir
from llvmlite.binding import Linkage
from numba.pycc import llvm_types as lt
from numba.core.cgutils import create_constant_array
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.registry import cpu_target
from numba.core.runtime import nrtdynmod
from numba.core import cgutils
@property
def module_init_definition(self):
    """
        Return the name and signature of the module's initialization function.
        """
    signature = ir.FunctionType(lt._pyobject_head_p, ())
    return (signature, 'PyInit_' + self.module_name)