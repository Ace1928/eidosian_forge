import re
import atexit
import ctypes
import os
import sys
import inspect
import platform
import numpy as _np
from . import libinfo
def c2pyerror(err_msg):
    """Translate C API error message to python style.

    Parameters
    ----------
    err_msg : str
        The error message.

    Returns
    -------
    new_msg : str
        Translated message.

    err_type : str
        Detected error type.
    """
    arr = err_msg.split('\n')
    if arr[-1] == '':
        arr.pop()
    err_type = _find_error_type(arr[0])
    trace_mode = False
    stack_trace = []
    message = []
    for line in arr:
        if trace_mode:
            if line.startswith('  '):
                stack_trace.append(line)
            else:
                trace_mode = False
        if not trace_mode:
            if line.startswith('Stack trace'):
                trace_mode = True
            else:
                message.append(line)
    out_msg = ''
    if stack_trace:
        out_msg += 'Traceback (most recent call last):\n'
        out_msg += '\n'.join(reversed(stack_trace)) + '\n'
    out_msg += '\n'.join(message)
    return (out_msg, err_type)