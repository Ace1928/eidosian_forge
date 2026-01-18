import warnings
import functools
import locale
import weakref
import ctypes
import html
import textwrap
import llvmlite.binding as ll
import llvmlite.ir as llvmir
from abc import abstractmethod, ABCMeta
from numba.core import utils, config, cgutils
from numba.core.llvm_bindings import create_pass_manager_builder
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.core.runtime import rtsys
from numba.core.compiler_lock import require_global_compiler_lock
from numba.core.errors import NumbaInvalidConfigWarning
from numba.misc.inspection import disassemble_elf_to_cfg
from numba.misc.llvm_pass_timings import PassTimingsCollection
def set_env(self, env_name, env):
    """Set the environment address.

        Update the GlobalVariable named *env_name* to the address of *env*.
        """
    gvaddr = self._engine.get_global_value_address(env_name)
    envptr = (ctypes.c_void_p * 1).from_address(gvaddr)
    envptr[0] = ctypes.c_void_p(id(env))