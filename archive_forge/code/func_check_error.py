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
def check_error(self, error, msg, exit=False):
    if error:
        exc = NvvmError(msg, RESULT_CODE_NAMES[error])
        if exit:
            print(exc)
            sys.exit(1)
        else:
            raise exc