import fnmatch
import functools
import io
import ntpath
import os
import posixpath
import re
import sys
import warnings
from _collections_abc import Sequence
from errno import ENOENT, ENOTDIR, EBADF, ELOOP
from operator import attrgetter
from stat import S_ISDIR, S_ISLNK, S_ISREG, S_ISSOCK, S_ISBLK, S_ISCHR, S_ISFIFO
from urllib.parse import quote_from_bytes as urlquote_from_bytes
def expanduser(self):
    """ Return a new path with expanded ~ and ~user constructs
        (as returned by os.path.expanduser)
        """
    if not (self._drv or self._root) and self._parts and (self._parts[0][:1] == '~'):
        homedir = os.path.expanduser(self._parts[0])
        if homedir[:1] == '~':
            raise RuntimeError('Could not determine home directory.')
        return self._from_parts([homedir] + self._parts[1:])
    return self