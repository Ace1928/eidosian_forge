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
def parse_parts(self, parts):
    parsed = []
    sep = self.sep
    altsep = self.altsep
    drv = root = ''
    it = reversed(parts)
    for part in it:
        if not part:
            continue
        if altsep:
            part = part.replace(altsep, sep)
        drv, root, rel = self.splitroot(part)
        if sep in rel:
            for x in reversed(rel.split(sep)):
                if x and x != '.':
                    parsed.append(sys.intern(x))
        elif rel and rel != '.':
            parsed.append(sys.intern(rel))
        if drv or root:
            if not drv:
                for part in it:
                    if not part:
                        continue
                    if altsep:
                        part = part.replace(altsep, sep)
                    drv = self.splitroot(part)[0]
                    if drv:
                        break
            break
    if drv or root:
        parsed.append(drv + root)
    parsed.reverse()
    return (drv, root, parsed)