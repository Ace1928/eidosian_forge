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
def get_arch_option(major, minor):
    """Matches with the closest architecture option
    """
    if config.FORCE_CUDA_CC:
        arch = config.FORCE_CUDA_CC
    else:
        arch = find_closest_arch((major, minor))
    return 'compute_%d%d' % arch