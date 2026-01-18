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
@_uncommitted()
def lzc_rename(source, target):
    """
    Rename the ZFS dataset.

    :param source name: the current name of the dataset to rename.
    :param target name: the new name of the dataset.
    :raises NameInvalid: if either the source or target name is invalid.
    :raises NameTooLong: if either the source or target name is too long.
    :raises NameTooLong: if a snapshot of the source would get a too long
                         name after renaming.
    :raises FilesystemNotFound: if the source does not exist.
    :raises FilesystemNotFound: if the target's parent does not exist.
    :raises FilesystemExists: if the target already exists.
    :raises PoolsDiffer: if the source and target belong to different pools.
    """
    ret = _lib.lzc_rename(source, target, _ffi.NULL, _ffi.NULL)
    errors.lzc_rename_translate_error(ret, source, target)