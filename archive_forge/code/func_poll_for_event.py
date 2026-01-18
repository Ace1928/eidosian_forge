from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def poll_for_event(self):
    e = lib.xcb_poll_for_event(self._conn)
    self.invalid()
    if e != ffi.NULL:
        return self.hoist_event(e)
    else:
        return None