import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def device_node(self):
    """
        Absolute path to the device node of this device as unicode string or
        ``None``, if this device doesn't have a device node.  The path
        includes the device directory (see :attr:`Context.device_path`).

        This path always points to the actual device node associated with
        this device, and never to any symbolic links to this device node.
        See :attr:`device_links` to get a list of symbolic links to this
        device node.

        .. warning::

           For devices created with :meth:`from_device_file()`, the value of
           this property is not necessary equal to the ``filename`` given to
           :meth:`from_device_file()`.
        """
    node = self._libudev.udev_device_get_devnode(self)
    return ensure_unicode_string(node) if node is not None else None