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
def get_prop_list_entry_p(self, prop_list_p, prop_list_sz, property_name):
    prop_list_p = ctypes.cast(prop_list_p, ctypes.POINTER(ctypes.c_ubyte * prop_list_sz))
    wb_prop_name = bytearray(ctypes.create_unicode_buffer(property_name))
    prop_list_addr = ctypes.addressof(prop_list_p.contents)
    prop_name_pos = bytearray(prop_list_p.contents).find(wb_prop_name)
    if prop_name_pos == -1:
        raise exceptions.ClusterPropertyListEntryNotFound(property_name=property_name)
    prop_name_len_pos = prop_name_pos - ctypes.sizeof(wintypes.DWORD)
    prop_name_len_addr = prop_list_addr + prop_name_len_pos
    prop_name_len = self._dword_align(wintypes.DWORD.from_address(prop_name_len_addr).value)
    prop_addr = prop_name_len_addr + prop_name_len + ctypes.sizeof(wintypes.DWORD)
    if prop_addr + ctypes.sizeof(wintypes.DWORD * 3) > prop_list_addr + prop_list_sz:
        raise exceptions.ClusterPropertyListParsingError()
    prop_entry = {'syntax': wintypes.DWORD.from_address(prop_addr).value, 'length': wintypes.DWORD.from_address(prop_addr + ctypes.sizeof(wintypes.DWORD)).value, 'val_p': ctypes.c_void_p(prop_addr + 2 * ctypes.sizeof(wintypes.DWORD))}
    return prop_entry