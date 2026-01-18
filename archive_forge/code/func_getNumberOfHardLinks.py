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
def getNumberOfHardLinks(self) -> int:
    """
        Retrieves the number of hard links to the file.

        This count keeps track of how many directories have entries for this
        file. If the count is ever decremented to zero then the file itself is
        discarded as soon as no process still holds it open.  Symbolic links
        are not counted in the total.

        @raise NotImplementedError: if the platform is Windows, since Windows
            doesn't maintain a link count for directories, and L{os.stat} does
            not set C{st_nlink} on Windows anyway.
        @return: the number of hard links to the file
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
    return st.st_nlink