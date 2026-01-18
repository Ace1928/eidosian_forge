import functools
import sys
import os
import tokenize
def getlines(filename, module_globals=None):
    """Get the lines for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""
    if filename in cache:
        entry = cache[filename]
        if len(entry) != 1:
            return cache[filename][2]
    try:
        return updatecache(filename, module_globals)
    except MemoryError:
        clearcache()
        return []