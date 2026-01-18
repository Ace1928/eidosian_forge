from __future__ import print_function, unicode_literals
import typing
from .errors import ResourceNotFound, ResourceReadOnly
from .info import Info
from .mode import check_writable
from .path import abspath, normpath, split
from .wrapfs import WrapFS
def cache_directory(fs):
    """Make a filesystem that caches directory information.

    Arguments:
        fs (FS): A filesystem instance.

    Returns:
        FS: A filesystem that caches results of `~FS.scandir`, `~FS.isdir`
        and other methods which read directory information.

    """
    return WrapCachedDir(fs)