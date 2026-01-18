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
def lzc_set_props(name, prop, val):
    """
    Set properties of the ZFS dataset.

    :param bytes name: the name of the dataset.
    :param bytes prop: the name of the property.
    :param Any val: the value of the property.
    :raises NameInvalid: if the dataset name is invalid.
    :raises NameTooLong: if the dataset name is too long.
    :raises DatasetNotFound: if the dataset does not exist.
    :raises NoSpace: if the property controls a quota and the values is
                     too small for that quota.
    :raises PropertyInvalid: if one or more of the specified properties is invalid
                             or has an invalid type or value.

    This function can be used on snapshots to set user defined properties.

    .. note::
        An attempt to set a readonly / statistic property is ignored
        without reporting any error.
    """
    props = {prop: val}
    props_nv = nvlist_in(props)
    ret = _lib.lzc_set_props(name, props_nv, _ffi.NULL, _ffi.NULL)
    errors.lzc_set_prop_translate_error(ret, name, prop, val)