from ctypes import (
import ctypes
from ctypes.util import find_library
import logging
import mmap
import os
import sysconfig
from .exception import ArchiveError
def check_null(ret, func, args):
    if ret is None:
        raise ArchiveError(func.__name__ + ' returned NULL')
    return ret