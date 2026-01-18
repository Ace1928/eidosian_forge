from __future__ import annotations
import atexit
from contextlib import contextmanager
import fnmatch
import importlib.util
import io
import os
from os.path import abspath
from os.path import dirname
from os.path import exists
from os.path import isabs
from os.path import isdir
from os.path import isfile
from os.path import islink
from os.path import normpath
import posixpath
from stat import S_ISDIR
from stat import S_ISLNK
from stat import S_ISREG
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Literal
from typing import overload
from typing import TYPE_CHECKING
import uuid
import warnings
from . import error
def relto(self, relpath):
    """Return a string which is the relative part of the path
        to the given 'relpath'.
        """
    if not isinstance(relpath, (str, LocalPath)):
        raise TypeError(f'{relpath!r}: not a string or path object')
    strrelpath = str(relpath)
    if strrelpath and strrelpath[-1] != self.sep:
        strrelpath += self.sep
    strself = self.strpath
    if sys.platform == 'win32' or getattr(os, '_name', None) == 'nt':
        if os.path.normcase(strself).startswith(os.path.normcase(strrelpath)):
            return strself[len(strrelpath):]
    elif strself.startswith(strrelpath):
        return strself[len(strrelpath):]
    return ''