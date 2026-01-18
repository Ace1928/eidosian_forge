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
def init_digraph(name, fname, fontsize):
    cmax = 200
    if len(fname) > cmax:
        wstr = f'CFG output filename "{fname}" exceeds maximum supported length, it will be truncated.'
        warnings.warn(wstr, NumbaInvalidConfigWarning)
        fname = fname[:cmax]
    f = gv.Digraph(name, filename=fname)
    f.attr(rankdir='TB')
    f.attr('node', shape='none', fontsize='%s' % str(fontsize))
    return f