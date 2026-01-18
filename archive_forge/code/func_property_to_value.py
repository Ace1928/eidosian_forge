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
@classmethod
def property_to_value(cls, prop):
    """
        :param prop: the property name as string
        :return: The numerical property value or ``None``

        This function is the equivalent to ``libevdev_property_from_name()``
        """
    v = cls._property_from_name(prop.encode('iso8859-1'))
    if v == -1:
        return None
    return v