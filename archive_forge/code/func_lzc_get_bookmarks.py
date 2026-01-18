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
def lzc_get_bookmarks(fsname, props=None):
    """
    Retrieve a listing of bookmarks for the given file system.

    :param bytes fsname: a name of the filesystem.
    :param props: a `list` of properties that will be returned for each bookmark.
    :type props: list of bytes
    :return: a `dict` that maps the bookmarks' short names to their properties.
    :rtype: dict of bytes:dict

    :raises FilesystemNotFound: if the filesystem is not found.

    The following are valid properties on bookmarks:

    guid : integer
        globally unique identifier of the snapshot the bookmark refers to
    createtxg : integer
        txg when the snapshot the bookmark refers to was created
    creation : integer
        timestamp when the snapshot the bookmark refers to was created

    Any other properties passed in ``props`` are ignored without reporting
    any error.
    Values in the returned dictionary map the names of the requested properties
    to their respective values.
    """
    bmarks = {}
    if props is None:
        props = []
    props_dict = {name: None for name in props}
    nvlist = nvlist_in(props_dict)
    with nvlist_out(bmarks) as bmarks_nvlist:
        ret = _lib.lzc_get_bookmarks(fsname, nvlist, bmarks_nvlist)
    errors.lzc_get_bookmarks_translate_error(ret, fsname, props)
    return bmarks