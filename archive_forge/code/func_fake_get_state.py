import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def fake_get_state(inst, group_handle, node_name_buff, node_name_len, error_ret_vals, error_on_nonzero_ret_val, ret_val_is_err_code):
    self.assertEqual(mock.sentinel.group_handle, group_handle)
    self.assertEqual([constants.CLUSTER_GROUP_STATE_UNKNOWN], error_ret_vals)
    self.assertFalse(error_on_nonzero_ret_val)
    self.assertFalse(ret_val_is_err_code)
    node_name_len_arg = ctypes.cast(node_name_len, wintypes.PDWORD).contents
    self.assertEqual(w_const.MAX_PATH, node_name_len_arg.value)
    node_name_arg = ctypes.cast(node_name_buff, ctypes.POINTER(ctypes.c_wchar * w_const.MAX_PATH)).contents
    node_name_arg.value = owner_node
    return mock.sentinel.group_state