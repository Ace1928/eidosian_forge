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
def select_from(self, parent_path):
    """Iterate over all child paths of `parent_path` matched by this
        selector.  This can contain parent_path itself."""
    path_cls = type(parent_path)
    is_dir = path_cls.is_dir
    exists = path_cls.exists
    scandir = path_cls._scandir
    if not is_dir(parent_path):
        return iter([])
    return self._select_from(parent_path, is_dir, exists, scandir)