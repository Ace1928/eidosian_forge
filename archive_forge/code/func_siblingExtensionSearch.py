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
def siblingExtensionSearch(self, *exts: OtherAnyStr) -> Optional[FilePath[OtherAnyStr]]:
    """
        Attempt to return a path with my name, given multiple possible
        extensions.

        Each extension in C{exts} will be tested and the first path which
        exists will be returned.  If no path exists, L{None} will be returned.
        If C{''} is in C{exts}, then if the file referred to by this path
        exists, C{self} will be returned.

        The extension '*' has a magic meaning, which means "any path that
        begins with C{self.path + '.'} is acceptable".
        """
    for ext in exts:
        if not ext and self.exists():
            return self.clonePath(self._getPathAsSameTypeAs(ext))
        p = self._getPathAsSameTypeAs(ext)
        star = _coerceToFilesystemEncoding(ext, '*')
        dot = _coerceToFilesystemEncoding(ext, '.')
        if ext == star:
            basedot = basename(p) + dot
            for fn in listdir(dirname(p)):
                if fn.startswith(basedot):
                    return self.clonePath(joinpath(dirname(p), fn))
        p2 = p + ext
        if exists(p2):
            return self.clonePath(p2)
    return None