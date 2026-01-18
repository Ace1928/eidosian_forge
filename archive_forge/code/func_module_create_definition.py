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
def module_create_definition(self):
    """
        Return the signature and name of the Python C API function to
        initialize the module.
        """
    signature = ir.FunctionType(lt._pyobject_head_p, (ir.PointerType(self.module_def_ty), lt._int32))
    name = 'PyModule_Create2'
    if lt._trace_refs_:
        name += 'TraceRefs'
    return (signature, name)