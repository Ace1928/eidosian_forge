import io
import os
import re
import tarfile
import tempfile
from .fnmatch import fnmatch
from ..constants import IS_WINDOWS_PLATFORM
def exclude_paths(root, patterns, dockerfile=None):
    """
    Given a root directory path and a list of .dockerignore patterns, return
    an iterator of all paths (both regular files and directories) in the root
    directory that do *not* match any of the patterns.

    All paths returned are relative to the root.
    """
    if dockerfile is None:
        dockerfile = 'Dockerfile'
    patterns.append('!' + dockerfile)
    pm = PatternMatcher(patterns)
    return set(pm.walk(root))