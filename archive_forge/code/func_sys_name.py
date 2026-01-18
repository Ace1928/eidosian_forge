import collections
import os
import re
import sys
from datetime import timedelta
from pyudev._errors import (
from pyudev._util import (
@property
def sys_name(self):
    """
        Device file name inside ``sysfs`` as unicode string.
        """
    return ensure_unicode_string(self._libudev.udev_device_get_sysname(self))