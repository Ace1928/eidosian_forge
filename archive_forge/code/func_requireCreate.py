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
def requireCreate(self, val: bool=True) -> None:
    """
        Sets the C{alwaysCreate} variable.

        @param val: C{True} or C{False}, indicating whether opening this path
            will be required to create the file or not.
        @type val: L{bool}

        @return: L{None}
        """
    self.alwaysCreate = val