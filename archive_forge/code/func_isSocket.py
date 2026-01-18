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
def isSocket(self) -> bool:
    """
        Returns whether the underlying path is a socket.

        @return: C{True} if it is a socket, C{False} otherwise
        @rtype: L{bool}
        @since: 11.1
        """
    st = self._statinfo
    if not st:
        self.restat(False)
        st = self._statinfo
        if not st:
            return False
    return S_ISSOCK(st.st_mode)