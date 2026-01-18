import functools
import os.path
from .errors import DistutilsFileError
from .py39compat import zip_strict
from ._functools import splat
def newer_group(sources, target, missing='error'):
    """
    Is target out-of-date with respect to any file in sources.

    Return True if 'target' is out-of-date with respect to any file
    listed in 'sources'. In other words, if 'target' exists and is newer
    than every file in 'sources', return False; otherwise return True.
    ``missing`` controls how to handle a missing source file:

    - error (default): allow the ``stat()`` call to fail.
    - ignore: silently disregard any missing source files.
    - newer: treat missing source files as "target out of date". This
      mode is handy in "dry-run" mode: it will pretend to carry out
      commands that wouldn't work because inputs are missing, but
      that doesn't matter because dry-run won't run the commands.
    """

    def missing_as_newer(source):
        return missing == 'newer' and (not os.path.exists(source))
    ignored = os.path.exists if missing == 'ignore' else None
    return any((missing_as_newer(source) or _newer(source, target) for source in filter(ignored, sources)))