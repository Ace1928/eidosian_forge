from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@rdevmajor.setter
def rdevmajor(self, value):
    ffi.entry_set_rdevmajor(self._entry_p, value)