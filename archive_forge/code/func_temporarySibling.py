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
def temporarySibling(self, extension: Optional[OtherAnyStr]=None) -> FilePath[OtherAnyStr]:
    """
        Construct a path referring to a sibling of this path.

        The resulting path will be unpredictable, so that other subprocesses
        should neither accidentally attempt to refer to the same path before it
        is created, nor they should other processes be able to guess its name
        in advance.

        @param extension: A suffix to append to the created filename.  (Note
            that if you want an extension with a '.' you must include the '.'
            yourself.)
        @type extension: L{bytes} or L{unicode}

        @return: a path object with the given extension suffix, C{alwaysCreate}
            set to True.
        @rtype: L{FilePath} with a mode equal to the type of C{extension}
        """
    ext: OtherAnyStr
    if extension is None:
        ext = self.path[0:0]
    else:
        ext = extension
    ourPath = self._getPathAsSameTypeAs(ext)
    sib = self.sibling(_secureEnoughString(ourPath) + self.clonePath(ourPath).basename() + ext)
    sib.requireCreate()
    return sib