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
def pypkgpath(self):
    """Return the Python package path by looking for the last
        directory upwards which still contains an __init__.py.
        Return None if a pkgpath can not be determined.
        """
    pkgpath = None
    for parent in self.parts(reverse=True):
        if parent.isdir():
            if not parent.join('__init__.py').exists():
                break
            if not isimportable(parent.basename):
                break
            pkgpath = parent
    return pkgpath