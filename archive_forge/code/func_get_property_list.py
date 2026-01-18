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
def get_property_list(self, property_entries):
    prop_entries_sz = sum([ctypes.sizeof(entry) for entry in property_entries])

    class CLUSPROP_LIST(ctypes.Structure):
        _fields_ = [('count', wintypes.DWORD), ('entries_buff', ctypes.c_ubyte * prop_entries_sz)]
    prop_list = CLUSPROP_LIST(count=len(property_entries))
    pos = 0
    for prop_entry in property_entries:
        prop_entry_sz = ctypes.sizeof(prop_entry)
        prop_list.entries_buff[pos:prop_entry_sz + pos] = bytearray(prop_entry)
        pos += prop_entry_sz
    return prop_list