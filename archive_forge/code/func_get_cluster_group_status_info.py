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
def get_cluster_group_status_info(self, prop_list_p, prop_list_sz):
    return self.get_prop_list_entry_value(prop_list_p, prop_list_sz, w_const.CLUSREG_NAME_GRP_STATUS_INFORMATION, ctypes.c_ulonglong, w_const.CLUSPROP_SYNTAX_LIST_VALUE_ULARGE_INTEGER)