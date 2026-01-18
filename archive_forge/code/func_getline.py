import functools
import sys
import os
import tokenize
def getline(filename, lineno, module_globals=None):
    """Get a line for a Python source file from the cache.
    Update the cache if it doesn't contain an entry for this file already."""
    lines = getlines(filename, module_globals)
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1]
    return ''