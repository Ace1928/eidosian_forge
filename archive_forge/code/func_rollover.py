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
def rollover(self):
    if self._rolled:
        return
    file = self._file
    newfile = self._file = TemporaryFile(**self._TemporaryFileArgs)
    del self._TemporaryFileArgs
    pos = file.tell()
    if hasattr(newfile, 'buffer'):
        newfile.buffer.write(file.detach().getvalue())
    else:
        newfile.write(file.getvalue())
    newfile.seek(pos, 0)
    self._rolled = True