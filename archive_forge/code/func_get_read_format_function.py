from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def get_read_format_function(format_name):
    function_name = 'read_support_format_' + format_name
    func = globals().get(function_name)
    if func:
        return func
    try:
        return ffi(function_name, [c_archive_p], c_int, check_int)
    except AttributeError:
        raise ValueError('the read format %r is not available' % format_name)