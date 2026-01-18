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
def ccs_supported_by_ctk(ctk_version):
    try:
        min_cc, max_cc = CTK_SUPPORTED[ctk_version]
        return tuple([cc for cc in COMPUTE_CAPABILITIES if min_cc <= cc <= max_cc])
    except KeyError:
        return tuple([cc for cc in COMPUTE_CAPABILITIES if cc >= config.CUDA_DEFAULT_PTX_CC])