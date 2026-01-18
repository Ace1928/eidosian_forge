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
def join_parsed_parts(self, drv, root, parts, drv2, root2, parts2):
    """
        Join the two paths represented by the respective
        (drive, root, parts) tuples.  Return a new (drive, root, parts) tuple.
        """
    if root2:
        if not drv2 and drv:
            return (drv, root2, [drv + root2] + parts2[1:])
    elif drv2:
        if drv2 == drv or self.casefold(drv2) == self.casefold(drv):
            return (drv, root, parts + parts2[1:])
    else:
        return (drv, root, parts + parts2)
    return (drv2, root2, parts2)