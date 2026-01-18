from __future__ import annotations
import base64
import errno
import os
import sys
from os import listdir, stat, utime
from os.path import (
from stat import (
from typing import (
from zope.interface import Attribute, Interface, implementer
from typing_extensions import Literal
from twisted.python.compat import cmp, comparable
from twisted.python.runtime import platform
from twisted.python.util import FancyEqMixin
from twisted.python.win32 import (
def getDevice(self) -> int:
    """
        Retrieves the device containing the file.  The inode number and device
        number together uniquely identify the file, but the device number is
        not necessarily consistent across reboots or system crashes.

        @raise NotImplementedError: if the platform is Windows, since the
            device number would be 0 for all partitions on a Windows platform

        @return: a number representing the device
        @rtype: L{int}

        @since: 11.0
        """
    if platform.isWindows():
        raise NotImplementedError
    st = self._statinfo
    if not st:
        self.restat()
        st = self._statinfo
    assert st is not None
    return st.st_dev