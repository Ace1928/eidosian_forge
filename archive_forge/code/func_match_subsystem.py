from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_subsystem(self, subsystem, nomatch=False):
    """
        Include all devices, which are part of the given ``subsystem``.

        ``subsystem`` is either a unicode string or a byte string, containing
        the name of the subsystem.  If ``nomatch`` is ``True`` (default is
        ``False``), the match is inverted:  A device is only included if it is
        *not* part of the given ``subsystem``.

        Note that, if a device has no subsystem, it is not included either
        with value of ``nomatch`` True or with value of ``nomatch`` False.

        Return the instance again.
        """
    match = self._libudev.udev_enumerate_add_nomatch_subsystem if nomatch else self._libudev.udev_enumerate_add_match_subsystem
    match(self, ensure_byte_string(subsystem))
    return self