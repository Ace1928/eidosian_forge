from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_property(self, prop, value):
    """
        Include all devices, whose ``prop`` has the given ``value``.

        ``prop`` is either a unicode string or a byte string, containing
        the name of the property to match.  ``value`` is a property value,
        being one of the following types:

        - :func:`int`
        - :func:`bool`
        - A byte string
        - Anything convertable to a unicode string (including a unicode string
          itself)

        Return the instance again.
        """
    self._libudev.udev_enumerate_add_match_property(self, ensure_byte_string(prop), property_value_to_bytes(value))
    return self