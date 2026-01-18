from __future__ import print_function, unicode_literals
import typing
from . import errors
from .errors import DirectoryNotEmpty, ResourceNotFound
from .path import abspath, dirname, normpath, recursepath
def is_thread_safe(*filesystems):
    """Check if all filesystems are thread-safe.

    Arguments:
        filesystems (FS): Filesystems instances to check.

    Returns:
        bool: if all filesystems are thread safe.

    """
    return all((fs.getmeta().get('thread_safe', False) for fs in filesystems))