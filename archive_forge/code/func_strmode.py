from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@property
def strmode(self):
    """The file's mode as a string, e.g. '?rwxrwx---'"""
    return ffi.entry_strmode(self._entry_p).strip()