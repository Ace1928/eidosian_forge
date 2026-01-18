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
def stringify_option(k, v):
    k = k.replace('_', '-')
    if v is None:
        return f'-{k}'
    if isinstance(v, bool):
        v = int(v)
    return f'-{k}={v}'