from contextlib import contextmanager
from ctypes import create_string_buffer
from enum import IntEnum
import math
from . import ffi
def set_ctime(self, timestamp_sec, timestamp_nsec=0):
    """Kept for backward compatibility. `entry.ctime = ...` is supported now."""
    return ffi.entry_set_ctime(self._entry_p, timestamp_sec, timestamp_nsec)