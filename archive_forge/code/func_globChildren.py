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
def globChildren(self, pattern: OtherAnyStr) -> List[FilePath[OtherAnyStr]]:
    """
        Assuming I am representing a directory, return a list of FilePaths
        representing my children that match the given pattern.

        @param pattern: A glob pattern to use to match child paths.
        @type pattern: L{unicode} or L{bytes}

        @return: A L{list} of matching children.
        @rtype: L{list} of L{FilePath}, with the mode of C{pattern}'s type
        """
    sep = _coerceToFilesystemEncoding(pattern, os.sep)
    ourPath = self._getPathAsSameTypeAs(pattern)
    import glob
    path = ourPath[-1] == sep and ourPath + pattern or sep.join([ourPath, pattern])
    return [self.clonePath(p) for p in glob.glob(path)]