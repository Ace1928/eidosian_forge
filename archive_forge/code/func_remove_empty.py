from __future__ import print_function, unicode_literals
import typing
from . import errors
from .errors import DirectoryNotEmpty, ResourceNotFound
from .path import abspath, dirname, normpath, recursepath
def remove_empty(fs, path):
    """Remove all empty parents.

    Arguments:
        fs (FS): A filesystem instance.
        path (str): Path to a directory on the filesystem.

    """
    path = abspath(normpath(path))
    try:
        while path not in ('', '/'):
            fs.removedir(path)
            path = dirname(path)
    except DirectoryNotEmpty:
        pass