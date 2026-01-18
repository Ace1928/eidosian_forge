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
def move_cluster_group(self, group_handle, destination_node_handle, move_flags, property_list):
    prop_list_p = ctypes.byref(property_list) if property_list else None
    prop_list_sz = ctypes.sizeof(property_list) if property_list else 0
    self._run_and_check_output(clusapi.MoveClusterGroupEx, group_handle, destination_node_handle, move_flags, prop_list_p, prop_list_sz, ignored_error_codes=[w_const.ERROR_IO_PENDING])