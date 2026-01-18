import os
import abc
import codecs
import errno
import stat
import sys
from _thread import allocate_lock as Lock
import io
from io import (__all__, SEEK_SET, SEEK_CUR, SEEK_END)
from _io import FileIO
@property
def closefd(self):
    """True if the file descriptor will be closed by close()."""
    return self._closefd