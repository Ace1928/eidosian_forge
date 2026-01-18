from __future__ import absolute_import, division
import ctypes.util
import functools
import io
import platform
import struct
import weakref
from .ffi import ffi
@ensure_connected
def get_screen_pointers(self):
    """
        Returns the xcb_screen_t for every screen
        useful for other bindings
        """
    root_iter = lib.xcb_setup_roots_iterator(self._setup)
    screens = [root_iter.data]
    for i in range(self._setup.roots_len - 1):
        lib.xcb_screen_next(ffi.addressof(root_iter))
        screens.append(root_iter.data)
    return screens