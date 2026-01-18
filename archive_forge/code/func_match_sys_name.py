from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_sys_name(self, sys_name):
    """
        Include all devices with the given name.

        ``sys_name`` is a byte or unicode string containing the device name.

        Return the instance again.

        .. versionadded:: 0.8
        """
    self._libudev.udev_enumerate_add_match_sysname(self, ensure_byte_string(sys_name))
    return self