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
def get_cluster_group_state(self, group_handle):
    node_name_len = wintypes.DWORD(w_const.MAX_PATH)
    node_name_buff = (ctypes.c_wchar * node_name_len.value)()
    group_state = self._run_and_check_output(clusapi.GetClusterGroupState, group_handle, node_name_buff, ctypes.byref(node_name_len), error_ret_vals=[constants.CLUSTER_GROUP_STATE_UNKNOWN], error_on_nonzero_ret_val=False, ret_val_is_err_code=False)
    return {'state': group_state, 'owner_node': node_name_buff.value}