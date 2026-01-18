from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
def has_error(self):
    return lib.xcb_connection_has_error(self._conn)