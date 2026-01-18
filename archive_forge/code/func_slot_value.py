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
def slot_value(self, slot, event_code, new_value=None):
    """
        :param slot: the numeric slot number
        :param event_code: the ABS_<*> event code, either as integer or string
        :param new_value: optional, the value to set this slot to
        :return: the current value of the slot's code, or ``None`` if it doesn't
                 exist on this device
        """
    t, c = self._code('EV_ABS', event_code)
    if new_value is not None:
        self._set_slot_value(self._ctx, slot, c, new_value)
    v = self._get_slot_value(self._ctx, slot, c)
    return v