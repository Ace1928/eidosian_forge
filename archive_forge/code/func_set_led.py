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
def set_led(self, led, on):
    """
        :param led: the LED_<*> name
        :on: True to turn the LED on, False to turn it off
        """
    t, c = self._code('EV_LED', led)
    which = 3 if on else 4
    self._kernel_set_led_value(self._ctx, c, which)