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
def lzc_destroy_bookmarks(bookmarks):
    """
    Destroy bookmarks.

    :param bookmarks: a list of the bookmarks to be destroyed.
                      The bookmarks are specified as :file:`{fs}#{bmark}`.
    :type bookmarks: list of bytes

    :raises BookmarkDestructionFailure: if any of the bookmarks may not be destroyed.

    The bookmarks must all be in the same pool.
    Bookmarks that do not exist will be silently ignored.
    This also includes the case where the filesystem component of the bookmark
    name does not exist.
    However, an invalid bookmark name will cause :exc:`.NameInvalid` error
    reported in :attr:`SnapshotDestructionFailure.errors`.

    Either all bookmarks that existed are destroyed or an exception is raised.
    """
    errlist = {}
    bmarks_dict = {name: None for name in bookmarks}
    nvlist = nvlist_in(bmarks_dict)
    with nvlist_out(errlist) as errlist_nvlist:
        ret = _lib.lzc_destroy_bookmarks(nvlist, errlist_nvlist)
    errors.lzc_destroy_bookmarks_translate_errors(ret, errlist, bookmarks)