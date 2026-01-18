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
def samefile(self, other_path):
    """Return whether other_path is the same or not as this file
        (as returned by os.path.samefile()).
        """
    st = self.stat()
    try:
        other_st = other_path.stat()
    except AttributeError:
        other_st = self.__class__(other_path).stat()
    return os.path.samestat(st, other_st)