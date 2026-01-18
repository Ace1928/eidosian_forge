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
def event_to_name(cls, event_type, event_code=None, event_value=None):
    """
        :param event_type: the numerical event type value
        :param event_code: optional, the numerical event code value
        :param event_value: optional, the numerical event value
        :return: the event code name if a code is given otherwise the event
                 type name.

        This function is the equivalent to ``libevdev_event_value_get_name()``,
        ``libevdev_event_code_get_name()``, and ``libevdev_event_type_get_name()``
        """
    if event_code is not None and event_value is not None:
        name = cls._event_value_get_name(event_type, event_code, event_value)
    elif event_code is not None:
        name = cls._event_code_get_name(event_type, event_code)
    else:
        name = cls._event_type_get_name(event_type)
    if not name:
        return None
    return name.decode('iso8859-1')