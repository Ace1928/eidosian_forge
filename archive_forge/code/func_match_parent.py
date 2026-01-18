from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_parent(self, parent):
    """
        Include all devices on the subtree of the given ``parent`` device.

        The ``parent`` device itself is also included.

        ``parent`` is a :class:`~pyudev.Device`.

        Return the instance again.

        .. udevversion:: 172

        .. versionadded:: 0.13
        """
    self._libudev.udev_enumerate_add_match_parent(self, parent)
    return self