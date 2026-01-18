from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def get_setup(self):
    self._setup = lib.xcb_get_setup(self._conn)
    buf = CffiUnpacker(self._setup, known_max=8 + self._setup.length * 4)
    return _setup(buf)