import os
import sys
import ctypes
import time
from ctypes import c_int, POINTER
from ctypes.wintypes import LPCWSTR, HLOCAL
from subprocess import STDOUT, TimeoutExpired
from threading import Thread
from ._process_common import read_no_interrupt, process_handler, arg_split as py_arg_split
from . import py3compat
from .encoding import DEFAULT_ENCODING
def arg_split(commandline, posix=False, strict=True):
    """Split a command line's arguments in a shell-like manner.

        This is a special version for windows that use a ctypes call to CommandLineToArgvW
        to do the argv splitting. The posix parameter is ignored.

        If strict=False, process_common.arg_split(...strict=False) is used instead.
        """
    if commandline.strip() == '':
        return []
    if not strict:
        return py_arg_split(commandline, posix=posix, strict=strict)
    argvn = c_int()
    result_pointer = CommandLineToArgvW(py3compat.cast_unicode(commandline.lstrip()), ctypes.byref(argvn))
    result_array_type = LPCWSTR * argvn.value
    result = [arg for arg in result_array_type.from_address(ctypes.addressof(result_pointer.contents))]
    retval = LocalFree(result_pointer)
    return result