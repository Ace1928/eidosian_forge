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
def lzc_destroy_one(name):
    """
    Destroy the ZFS dataset.

    :param bytes name: the name of the dataset to destroy.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :raises FilesystemNotFound: if the dataset does not exist.
    """
    ret = _lib.lzc_destroy_one(name, _ffi.NULL)
    errors.lzc_destroy_translate_error(ret, name)