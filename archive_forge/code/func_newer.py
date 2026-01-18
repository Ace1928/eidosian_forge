import functools
import os.path
from .errors import DistutilsFileError
from .py39compat import zip_strict
from ._functools import splat
def newer(source, target):
    """
    Is source modified more recently than target.

    Returns True if 'source' is modified more recently than
    'target' or if 'target' does not exist.

    Raises DistutilsFileError if 'source' does not exist.
    """
    if not os.path.exists(source):
        raise DistutilsFileError("file '%s' does not exist" % os.path.abspath(source))
    return _newer(source, target)