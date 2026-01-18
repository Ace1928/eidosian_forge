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
def emit_type(self, tyobj):
    ret_val = str(tyobj)
    if 'int' in ret_val:
        if ret_val.endswith(('8', '16', '32', '64')):
            ret_val += '_t'
    return ret_val