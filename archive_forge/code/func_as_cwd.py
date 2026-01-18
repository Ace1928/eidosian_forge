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
@contextmanager
def as_cwd(self):
    """
        Return a context manager, which changes to the path's dir during the
        managed "with" context.
        On __enter__ it returns the old dir, which might be ``None``.
        """
    old = self.chdir()
    try:
        yield old
    finally:
        if old is not None:
            old.chdir()