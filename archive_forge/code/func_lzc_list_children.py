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
@_uncommitted(lzc_list)
def lzc_list_children(name):
    """
    List the children of the ZFS dataset.

    :param bytes name: the name of the dataset.
    :return: an iterator that produces the names of the children.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :raises DatasetNotFound: if the dataset does not exist.

    .. warning::
        If the dataset does not exist, then the returned iterator would produce
        no results and no error is reported.
        That case is indistinguishable from the dataset having no children.

        An attempt to list children of a snapshot is silently ignored as well.
    """
    children = []
    for entry in _list(name, recurse=1, types=['filesystem', 'volume']):
        child = entry['name']
        if child != name:
            children.append(child)
    return iter(children)