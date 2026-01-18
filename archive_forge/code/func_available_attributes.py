import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def available_attributes(self):
    """
        Yield the ``available`` attributes for the device.

        It is not guaranteed that a key in this list will have a value.
        It is not guaranteed that a key not in this list will not have a value.

        It is guaranteed that the keys in this list are the keys that libudev
        considers to be "available" attributes.

        If libudev version does not define udev_device_get_sysattr_list_entry()
        yields nothing.

        See rhbz#1267584.
        """
    if not hasattr(self._libudev, 'udev_device_get_sysattr_list_entry'):
        return
    attrs = self._libudev.udev_device_get_sysattr_list_entry(self.device)
    for attribute, _ in udev_list_iterate(self._libudev, attrs):
        yield ensure_unicode_string(attribute)