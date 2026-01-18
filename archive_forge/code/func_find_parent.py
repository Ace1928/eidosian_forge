import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
def find_parent(self, subsystem, device_type=None):
    """
        Find the parent device with the given ``subsystem`` and
        ``device_type``.

        ``subsystem`` is a byte or unicode string containing the name of the
        subsystem, in which to search for the parent.  ``device_type`` is a
        byte or unicode string holding the expected device type of the parent.
        It can be ``None`` (the default), which means, that no specific device
        type is expected.

        Return a parent :class:`Device` within the given ``subsystem`` and, if
        ``device_type`` is not ``None``, with the given ``device_type``, or
        ``None``, if this device has no parent device matching these
        constraints.

        .. versionadded:: 0.9
        """
    subsystem = ensure_byte_string(subsystem)
    if device_type is not None:
        device_type = ensure_byte_string(device_type)
    parent = self._libudev.udev_device_get_parent_with_subsystem_devtype(self, subsystem, device_type)
    if not parent:
        return None
    return Device(self.context, self._libudev.udev_device_ref(parent))