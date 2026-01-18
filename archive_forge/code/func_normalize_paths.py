from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
def normalize_paths(value, parent=os.curdir):
    """Parse a comma-separated list of paths.

    Return a list of absolute paths.
    """
    if not value:
        return []
    if isinstance(value, list):
        return value
    paths = []
    for path in value.split(','):
        path = path.strip()
        if '/' in path:
            path = os.path.abspath(os.path.join(parent, path))
        paths.append(path.rstrip('/'))
    return paths