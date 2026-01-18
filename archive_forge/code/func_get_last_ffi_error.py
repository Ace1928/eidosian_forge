import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def get_last_ffi_error():
    """Create error object given result of MXGetLastError.

    Returns
    -------
    err : object
        The error object based on the err_msg
    """
    c_err_msg = py_str(_LIB.MXGetLastError())
    py_err_msg, err_type = c2pyerror(c_err_msg)
    if err_type is not None and err_type.startswith('mxnet.error.'):
        err_type = err_type[10:]
    return error_types.get(err_type, MXNetError)(py_err_msg)