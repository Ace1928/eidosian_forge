import logging
import re
import sys
import warnings
from ctypes import (c_void_p, c_int, POINTER, c_char_p, c_size_t, byref,
import threading
from llvmlite import ir
from .error import NvvmError, NvvmSupportError, NvvmWarning
from .libs import get_libdevice, open_libdevice, open_cudalib
from numba.core import cgutils, config
def llvm140_to_70_ir(ir):
    """
    Convert LLVM 14.0 IR for LLVM 7.0.
    """
    buf = []
    for line in ir.splitlines():
        if line.startswith('attributes #'):
            m = re_attributes_def.match(line)
            attrs = m.group(1).split()
            attrs = ' '.join((a for a in attrs if a != 'willreturn'))
            line = line.replace(m.group(1), attrs)
        buf.append(line)
    return '\n'.join(buf)