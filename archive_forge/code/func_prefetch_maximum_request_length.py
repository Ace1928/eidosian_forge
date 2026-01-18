from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def prefetch_maximum_request_length(self):
    return lib.xcb_prefetch_maximum_request_length(self._conn)