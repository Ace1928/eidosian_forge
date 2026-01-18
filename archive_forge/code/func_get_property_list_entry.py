import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def get_property_list_entry(self, name, syntax, value):
    name_len = len(name) + 1
    val_sz = ctypes.sizeof(value)

    class CLUSPROP_LIST_ENTRY(ctypes.Structure):
        _fields_ = [('name', self._get_clusprop_value_struct(val_type=ctypes.c_wchar * name_len)), ('value', self._get_clusprop_value_struct(val_type=ctypes.c_ubyte * val_sz)), ('_endmark', wintypes.DWORD)]
    entry = CLUSPROP_LIST_ENTRY()
    entry.name.syntax = w_const.CLUSPROP_SYNTAX_NAME
    entry.name.length = name_len * ctypes.sizeof(ctypes.c_wchar)
    entry.name.value = name
    entry.value.syntax = syntax
    entry.value.length = val_sz
    entry.value.value[0:val_sz] = bytearray(value)
    entry._endmark = w_const.CLUSPROP_SYNTAX_ENDMARK
    return entry