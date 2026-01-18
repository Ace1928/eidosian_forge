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
def ensure(self, *args, **kwargs):
    """Ensure that an args-joined path exists (by default as
        a file). if you specify a keyword argument 'dir=True'
        then the path is forced to be a directory path.
        """
    p = self.join(*args)
    if kwargs.get('dir', 0):
        return p._ensuredirs()
    else:
        p.dirpath()._ensuredirs()
        if not p.check(file=1):
            p.open('wb').close()
        return p