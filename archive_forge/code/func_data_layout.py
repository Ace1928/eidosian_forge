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
@property
def data_layout(self):
    if (self._majorIR, self._minorIR) < (1, 8):
        return _datalayout_original
    else:
        return _datalayout_i128