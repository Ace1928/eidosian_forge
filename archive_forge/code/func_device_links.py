import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def device_links(self):
    """
        An iterator, which yields the absolute paths (including the device
        directory, see :attr:`Context.device_path`) of all symbolic links
        pointing to the :attr:`device_node` of this device.  The paths are
        unicode strings.

        UDev can create symlinks to the original device node (see
        :attr:`device_node`) inside the device directory.  This is often
        used to assign a constant, fixed device node to devices like
        removeable media, which technically do not have a constant device
        node, or to map a single device into multiple device hierarchies.
        The property provides access to all such symbolic links, which were
        created by UDev for this device.

        .. warning::

           Links are not necessarily resolved by
           :meth:`Devices.from_device_file()`. Hence do *not* rely on
           ``Devices.from_device_file(context, link).device_path ==
           device.device_path`` from any ``link`` in ``device.device_links``.
        """
    devlinks = self._libudev.udev_device_get_devlinks_list_entry(self)
    for name, _ in udev_list_iterate(self._libudev, devlinks):
        yield ensure_unicode_string(name)