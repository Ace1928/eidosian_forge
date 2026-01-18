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
def cluster_enum(self, enum_handle, index):
    item_sz = wintypes.DWORD(0)
    self._run_and_check_output(clusapi.ClusterEnumEx, enum_handle, index, None, ctypes.byref(item_sz), ignored_error_codes=[w_const.ERROR_MORE_DATA])
    item_buff = (ctypes.c_ubyte * item_sz.value)()
    self._run_and_check_output(clusapi.ClusterEnumEx, enum_handle, index, ctypes.byref(item_buff), ctypes.byref(item_sz))
    return ctypes.cast(item_buff, clusapi_def.PCLUSTER_ENUM_ITEM).contents