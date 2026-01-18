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
def segmentsFrom(self: _Self, ancestor: _Self) -> List[AnyStr]:
    """
        Return a list of segments between a child and its ancestor.

        For example, in the case of a path X representing /a/b/c/d and a path Y
        representing /a/b, C{Y.segmentsFrom(X)} will return C{['c',
        'd']}.

        @param ancestor: an instance of the same class as self, ostensibly an
        ancestor of self.

        @raise ValueError: If the C{ancestor} parameter is not actually an
        ancestor, i.e. a path for /x/y/z is passed as an ancestor for /a/b/c/d.

        @return: a list of strs
        """
    f = self
    p: _Self = f.parent()
    segments: List[AnyStr] = []
    while f != ancestor and p != f:
        segments[0:0] = [f.basename()]
        f = p
        p = p.parent()
    if f == ancestor and segments:
        return segments
    raise ValueError(f'{ancestor!r} not parent of {self!r}')