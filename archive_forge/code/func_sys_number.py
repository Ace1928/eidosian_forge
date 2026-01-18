import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def sys_number(self):
    """
        The trailing number of the :attr:`sys_name` as unicode string, or
        ``None``, if the device has no trailing number in its name.

        .. note::

           The number is returned as unicode string to preserve the exact
           format of the number, especially any leading zeros:

           >>> from pyudev import Context, Device
           >>> context = Context()
           >>> device = Devices.from_path(context, '/sys/devices/LNXSYSTM:00')
           >>> device.sys_number
           u'00'

           To work with numbers, explicitly convert them to ints:

           >>> int(device.sys_number)
           0

        .. versionadded:: 0.11
        """
    number = self._libudev.udev_device_get_sysnum(self)
    return ensure_unicode_string(number) if number is not None else None