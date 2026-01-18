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
def event_to_value(cls, event_type, event_code=None, event_value=None):
    """
        :param event_type: the event type as string
        :param event_code: optional, the event code as string
        :param event_value: optional, the numerical event value
        :return: the event code value if a code is given otherwise the event
                 type value.

        This function is the equivalent to ``libevdev_event_value_from_name()``,
        ``libevdev_event_code_from_name()`` and ``libevdev_event_type_from_name()``
        """
    if event_code is not None and event_value is not None:
        if not isinstance(event_type, int):
            event_type = cls.event_to_value(event_type)
        if not isinstance(event_code, int):
            event_code = cls.event_to_value(event_type, event_code)
        v = cls._event_value_from_name(event_type, event_code, event_value.encode('iso8859-1'))
    elif event_code is not None:
        if not isinstance(event_type, int):
            event_type = cls.event_to_value(event_type)
        v = cls._event_code_from_name(event_type, event_code.encode('iso8859-1'))
    else:
        v = cls._event_type_from_name(event_type.encode('iso8859-1'))
    if v == -1:
        return None
    return v