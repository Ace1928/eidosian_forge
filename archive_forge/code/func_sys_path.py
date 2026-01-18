import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def sys_path(self):
    """
        Absolute path of this device in ``sysfs`` including the ``sysfs``
        mount point as unicode string.
        """
    return ensure_unicode_string(self._libudev.udev_device_get_syspath(self))