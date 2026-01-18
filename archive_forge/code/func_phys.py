import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
@phys.setter
def phys(self, phys):
    if phys is None:
        phys = ''
    return self._set_phys(self._ctx, phys.encode('iso8859-1'))