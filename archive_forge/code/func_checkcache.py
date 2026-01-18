import functools
import sys
import os
import tokenize
def checkcache(filename=None):
    """Discard cache entries that are out of date.
    (This is not checked upon each call!)"""
    if filename is None:
        filenames = list(cache.keys())
    elif filename in cache:
        filenames = [filename]
    else:
        return
    for filename in filenames:
        entry = cache[filename]
        if len(entry) == 1:
            continue
        size, mtime, lines, fullname = entry
        if mtime is None:
            continue
        try:
            stat = os.stat(fullname)
        except OSError:
            cache.pop(filename, None)
            continue
        if size != stat.st_size or mtime != stat.st_mtime:
            cache.pop(filename, None)