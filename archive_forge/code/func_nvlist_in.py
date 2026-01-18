import numbers
from collections import namedtuple
from contextlib import contextmanager
from .bindings import libnvpair
from .ctypes import _type_to_suffix
def nvlist_in(props):
    """
    This function converts a python dictionary to a C nvlist_t
    and provides automatic memory management for the latter.

    :param dict props: the dictionary to be converted.
    :return: an FFI CData object representing the nvlist_t pointer.
    :rtype: CData
    """
    nvlistp = _ffi.new('nvlist_t **')
    res = _lib.nvlist_alloc(nvlistp, 1, 0)
    if res != 0:
        raise MemoryError('nvlist_alloc failed')
    nvlist = _ffi.gc(nvlistp[0], _lib.nvlist_free)
    _dict_to_nvlist(props, nvlist)
    return nvlist