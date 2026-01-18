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
def get_supported_ccs():
    try:
        from numba.cuda.cudadrv.runtime import runtime
        cudart_version = runtime.get_version()
    except:
        _supported_cc = ()
        return _supported_cc
    min_cudart = min(CTK_SUPPORTED)
    if cudart_version < min_cudart:
        _supported_cc = ()
        ctk_ver = f'{cudart_version[0]}.{cudart_version[1]}'
        unsupported_ver = f'CUDA Toolkit {ctk_ver} is unsupported by Numba - {min_cudart[0]}.{min_cudart[1]} is the minimum required version.'
        warnings.warn(unsupported_ver)
        return _supported_cc
    _supported_cc = ccs_supported_by_ctk(cudart_version)
    return _supported_cc