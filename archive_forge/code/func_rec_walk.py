import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def rec_walk(current_dir):
    for f in os.listdir(current_dir):
        fpath = os.path.join(os.path.relpath(current_dir, root), f)
        if fpath.startswith('.' + os.path.sep):
            fpath = fpath[2:]
        match = self.matches(fpath)
        if not match:
            yield fpath
        cur = os.path.join(root, fpath)
        if not os.path.isdir(cur) or os.path.islink(cur):
            continue
        if match:
            skip = True
            for pat in self.patterns:
                if not pat.exclusion:
                    continue
                if pat.cleaned_pattern.startswith(normalize_slashes(fpath)):
                    skip = False
                    break
            if skip:
                continue
        yield from rec_walk(cur)