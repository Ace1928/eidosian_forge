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
def get_ir_version(self):
    majorIR = c_int()
    minorIR = c_int()
    majorDbg = c_int()
    minorDbg = c_int()
    err = self.nvvmIRVersion(byref(majorIR), byref(minorIR), byref(majorDbg), byref(minorDbg))
    self.check_error(err, 'Failed to get IR version.')
    return (majorIR.value, minorIR.value, majorDbg.value, minorDbg.value)