from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@rdev.setter
def rdev(self, value):
    if isinstance(value, tuple):
        ffi.entry_set_rdevmajor(self._entry_p, value[0])
        ffi.entry_set_rdevminor(self._entry_p, value[1])
    else:
        ffi.entry_set_rdev(self._entry_p, value)