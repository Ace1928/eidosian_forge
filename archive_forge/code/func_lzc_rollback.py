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
def lzc_rollback(name):
    """
    Roll back a filesystem or volume to its most recent snapshot.

    :param bytes name: a name of the dataset to be rolled back.
    :return: a name of the most recent snapshot.
    :rtype: bytes

    :raises FilesystemNotFound: if the dataset does not exist.
    :raises SnapshotNotFound: if the dataset does not have any snapshots.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    """
    snapnamep = _ffi.new('char[]', MAXNAMELEN + 1)
    ret = _lib.lzc_rollback(name, snapnamep, MAXNAMELEN + 1)
    errors.lzc_rollback_translate_error(ret, name)
    return _ffi.string(snapnamep)