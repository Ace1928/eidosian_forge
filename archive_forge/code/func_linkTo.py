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
def linkTo(self, linkFilePath: FilePath[AnyStr]) -> None:
    """
        Creates a symlink to self to at the path in the L{FilePath}
        C{linkFilePath}.

        Only works on posix systems due to its dependence on
        L{os.symlink}.  Propagates L{OSError}s up from L{os.symlink} if
        C{linkFilePath.parent()} does not exist, or C{linkFilePath} already
        exists.

        @param linkFilePath: a FilePath representing the link to be created.
        @type linkFilePath: L{FilePath}
        """
    os.symlink(self.path, linkFilePath.path)