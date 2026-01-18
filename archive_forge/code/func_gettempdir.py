import functools as _functools
import warnings as _warnings
import io as _io
import os as _os
import shutil as _shutil
import stat as _stat
import errno as _errno
from random import Random as _Random
import sys as _sys
import types as _types
import weakref as _weakref
import _thread
def gettempdir():
    """Returns tempfile.tempdir as str."""
    return _os.fsdecode(_gettempdir())