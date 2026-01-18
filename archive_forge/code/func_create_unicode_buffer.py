import os as _os, sys as _sys
import types as _types
from _ctypes import Union, Structure, Array
from _ctypes import _Pointer
from _ctypes import CFuncPtr as _CFuncPtr
from _ctypes import __version__ as _ctypes_version
from _ctypes import RTLD_LOCAL, RTLD_GLOBAL
from _ctypes import ArgumentError
from struct import calcsize as _calcsize
from _ctypes import FUNCFLAG_CDECL as _FUNCFLAG_CDECL, \
from _ctypes import sizeof, byref, addressof, alignment, resize
from _ctypes import get_errno, set_errno
from _ctypes import _SimpleCData
from _ctypes import POINTER, pointer, _pointer_type_cache
from _ctypes import _memmove_addr, _memset_addr, _string_at_addr, _cast_addr
from ctypes._endian import BigEndianStructure, LittleEndianStructure
from ctypes._endian import BigEndianUnion, LittleEndianUnion
def create_unicode_buffer(init, size=None):
    """create_unicode_buffer(aString) -> character array
    create_unicode_buffer(anInteger) -> character array
    create_unicode_buffer(aString, anInteger) -> character array
    """
    if isinstance(init, str):
        if size is None:
            if sizeof(c_wchar) == 2:
                size = sum((2 if ord(c) > 65535 else 1 for c in init)) + 1
            else:
                size = len(init) + 1
        _sys.audit('ctypes.create_unicode_buffer', init, size)
        buftype = c_wchar * size
        buf = buftype()
        buf.value = init
        return buf
    elif isinstance(init, int):
        _sys.audit('ctypes.create_unicode_buffer', None, init)
        buftype = c_wchar * init
        buf = buftype()
        return buf
    raise TypeError(init)