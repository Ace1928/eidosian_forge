from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
def set_birthtime(self, timestamp_sec, timestamp_nsec=0):
    """Kept for backward compatibility. `entry.birthtime = ...` is supported now."""
    return ffi.entry_set_birthtime(self._entry_p, timestamp_sec, timestamp_nsec)