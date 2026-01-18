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
def lzc_create(name, ds_type='zfs', props=None):
    """
    Create a ZFS filesystem or a ZFS volume ("zvol").

    :param bytes name: a name of the dataset to be created.
    :param str ds_type: the type of the dataset to be create, currently supported
                        types are "zfs" (the default) for a filesystem
                        and "zvol" for a volume.
    :param props: a `dict` of ZFS dataset property name-value pairs (empty by default).
    :type props: dict of bytes:Any

    :raises FilesystemExists: if a dataset with the given name already exists.
    :raises ParentNotFound: if a parent dataset of the requested dataset does not exist.
    :raises PropertyInvalid: if one or more of the specified properties is invalid
                             or has an invalid type or value.
    :raises NameInvalid: if the name is not a valid dataset name.
    :raises NameTooLong: if the name is too long.
    """
    if props is None:
        props = {}
    if ds_type == 'zfs':
        ds_type = _lib.DMU_OST_ZFS
    elif ds_type == 'zvol':
        ds_type = _lib.DMU_OST_ZVOL
    else:
        raise exceptions.DatasetTypeInvalid(ds_type)
    nvlist = nvlist_in(props)
    ret = _lib.lzc_create(name, ds_type, nvlist)
    errors.lzc_create_translate_error(ret, name, ds_type, props)