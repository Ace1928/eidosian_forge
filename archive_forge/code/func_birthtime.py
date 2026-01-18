from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
@birthtime.setter
def birthtime(self, value):
    if value is None:
        ffi.entry_unset_birthtime(self._entry_p)
    elif isinstance(value, int):
        self.set_birthtime(value)
    elif isinstance(value, tuple):
        self.set_birthtime(*value)
    else:
        seconds, fraction = math.modf(value)
        self.set_birthtime(int(seconds), int(fraction * 1000000000))