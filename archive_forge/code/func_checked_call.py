from ctypes import byref, c_char, c_char_p, c_int, c_size_t, c_void_p, POINTER
from enum import IntEnum
from numba.core import config
from numba.cuda.cudadrv.error import (NvrtcError, NvrtcCompilationError,
import functools
import os
import threading
import warnings
@functools.wraps(func)
def checked_call(*args, func=func, name=name):
    error = func(*args)
    if error == NvrtcResult.NVRTC_ERROR_COMPILATION:
        raise NvrtcCompilationError()
    elif error != NvrtcResult.NVRTC_SUCCESS:
        try:
            error_name = NvrtcResult(error).name
        except ValueError:
            error_name = f'Unknown nvrtc_result (error code: {error})'
        msg = f'Failed to call {name}: {error_name}'
        raise NvrtcError(msg)