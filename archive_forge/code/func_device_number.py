import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def device_number(self):
    """
        The device number of the associated device as integer, or ``0``, if no
        device number is associated.

        Use :func:`os.major` and :func:`os.minor` to decompose the device
        number into its major and minor number:

        >>> import os
        >>> from pyudev import Context, Device
        >>> context = Context()
        >>> sda = Devices.from_name(context, 'block', 'sda')
        >>> sda.device_number
        2048L
        >>> (os.major(sda.device_number), os.minor(sda.device_number))
        (8, 0)

        For devices with an associated :attr:`device_node`, this is the same as
        the ``st_rdev`` field of the stat result of the :attr:`device_node`:

        >>> os.stat(sda.device_node).st_rdev
        2048

        .. versionadded:: 0.11
        """
    return self._libudev.udev_device_get_devnum(self)