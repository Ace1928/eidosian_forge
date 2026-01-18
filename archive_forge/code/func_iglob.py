import contextlib
import os
import re
import fnmatch
import itertools
import stat
import sys
def iglob(pathname, *, root_dir=None, dir_fd=None, recursive=False, include_hidden=False):
    """Return an iterator which yields the paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards a la
    fnmatch. However, unlike fnmatch, filenames starting with a
    dot are special cases that are not matched by '*' and '?'
    patterns.

    If recursive is true, the pattern '**' will match any files and
    zero or more directories and subdirectories.
    """
    sys.audit('glob.glob', pathname, recursive)
    sys.audit('glob.glob/2', pathname, recursive, root_dir, dir_fd)
    if root_dir is not None:
        root_dir = os.fspath(root_dir)
    else:
        root_dir = pathname[:0]
    it = _iglob(pathname, root_dir, dir_fd, recursive, False, include_hidden=include_hidden)
    if not pathname or (recursive and _isrecursive(pathname[:2])):
        try:
            s = next(it)
            if s:
                it = itertools.chain((s,), it)
        except StopIteration:
            pass
    return it