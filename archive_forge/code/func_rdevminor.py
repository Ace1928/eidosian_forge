from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@rdevminor.setter
def rdevminor(self, value):
    ffi.entry_set_rdevminor(self._entry_p, value)