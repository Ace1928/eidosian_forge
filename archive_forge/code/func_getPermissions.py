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
def getPermissions(self) -> Permissions:
    """
        Returns the permissions of the file.  Should also work on Windows,
        however, those permissions may not be what is expected in Windows.

        @return: the permissions for the file
        @rtype: L{Permissions}
        @since: 11.1
        """
    st = self._statinfo
    if not st:
        self.restat()
        st = self._statinfo
    assert st is not None
    return Permissions(S_IMODE(st.st_mode))