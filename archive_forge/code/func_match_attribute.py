from pyudev._ctypeslib.libudev import ERROR_CHECKERS, SIGNATURES
from pyudev._ctypeslib.utils import load_ctypes_library
from pyudev._errors import DeviceNotFoundAtPathError
from pyudev._util import (
from pyudev.device import Devices
def match_attribute(self, attribute, value, nomatch=False):
    """
        Include all devices, whose ``attribute`` has the given ``value``.

        ``attribute`` is either a unicode string or a byte string, containing
        the name of a sys attribute to match.  ``value`` is an attribute value,
        being one of the following types:

        - :func:`int`,
        - :func:`bool`
        - A byte string
        - Anything convertable to a unicode string (including a unicode string
          itself)

        If ``nomatch`` is ``True`` (default is ``False``), the match is
        inverted:  A device is include if the ``attribute`` does *not* match
        the given ``value``.

        .. note::

           If ``nomatch`` is ``True``, devices which do not have the given
           ``attribute`` at all are also included.  In other words, with
           ``nomatch=True`` the given ``attribute`` is *not* guaranteed to
           exist on all returned devices.

        Return the instance again.
        """
    match = self._libudev.udev_enumerate_add_match_sysattr if not nomatch else self._libudev.udev_enumerate_add_nomatch_sysattr
    match(self, ensure_byte_string(attribute), property_value_to_bytes(value))
    return self