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
def set_clock_id(self, clock):
    """
        :param clock: time.CLOCK_MONOTONIC
        :return: a negative errno on failure or 0 on success.
        """
    return self._set_clock_id(self._ctx, clock)