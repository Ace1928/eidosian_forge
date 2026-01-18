import os
import errno
from distutils.errors import DistutilsFileError, DistutilsInternalError
from distutils import log
def ensure_relative(path):
    """Take the full path 'path', and make it a relative path.

    This is useful to make 'path' the second argument to os.path.join().
    """
    drive, path = os.path.splitdrive(path)
    if path[0:1] == os.sep:
        path = drive + path[1:]
    return path