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
def relative_to(self, *other):
    """Return the relative path to another path identified by the passed
        arguments.  If the operation is not possible (because this is not
        a subpath of the other path), raise ValueError.
        """
    if not other:
        raise TypeError('need at least one argument')
    parts = self._parts
    drv = self._drv
    root = self._root
    if root:
        abs_parts = [drv, root] + parts[1:]
    else:
        abs_parts = parts
    to_drv, to_root, to_parts = self._parse_args(other)
    if to_root:
        to_abs_parts = [to_drv, to_root] + to_parts[1:]
    else:
        to_abs_parts = to_parts
    n = len(to_abs_parts)
    cf = self._flavour.casefold_parts
    if root or drv if n == 0 else cf(abs_parts[:n]) != cf(to_abs_parts):
        formatted = self._format_parsed_parts(to_drv, to_root, to_parts)
        raise ValueError('{!r} is not in the subpath of {!r} OR one path is relative and the other is absolute.'.format(str(self), str(formatted)))
    return self._from_parsed_parts('', root if n == 1 else '', abs_parts[n:])