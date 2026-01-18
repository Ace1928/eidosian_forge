import errno
import functools
import fcntl
import os
import struct
import threading
from . import exceptions
from . import _error_translation as errors
from .bindings import libzfs_core
from ._constants import MAXNAMELEN
from .ctypes import int32_t
from ._nvlist import nvlist_in, nvlist_out
def lzc_bookmark(bookmarks):
    """
    Create bookmarks.

    :param bookmarks: a dict that maps names of wanted bookmarks to names of existing snapshots.
    :type bookmarks: dict of bytes to bytes

    :raises BookmarkFailure: if any of the bookmarks can not be created for any reason.

    The bookmarks `dict` maps from name of the bookmark (e.g. :file:`{pool}/{fs}#{bmark}`) to
    the name of the snapshot (e.g. :file:`{pool}/{fs}@{snap}`).  All the bookmarks and
    snapshots must be in the same pool.
    """
    errlist = {}
    nvlist = nvlist_in(bookmarks)
    with nvlist_out(errlist) as errlist_nvlist:
        ret = _lib.lzc_bookmark(nvlist, errlist_nvlist)
    errors.lzc_bookmark_translate_errors(ret, errlist, bookmarks)