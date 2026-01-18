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
@ddt.data(mock.sentinel.prop_list, None)
def test_move_cluster_group(self, prop_list):
    self._mock_ctypes()
    expected_prop_list_arg = self._ctypes.byref(prop_list) if prop_list else None
    expected_prop_list_sz = self._ctypes.sizeof(prop_list) if prop_list else 0
    self._clusapi_utils.move_cluster_group(mock.sentinel.group_handle, mock.sentinel.dest_node_handle, mock.sentinel.move_flags, prop_list)
    self._mock_run.assert_called_once_with(self._clusapi.MoveClusterGroupEx, mock.sentinel.group_handle, mock.sentinel.dest_node_handle, mock.sentinel.move_flags, expected_prop_list_arg, expected_prop_list_sz, ignored_error_codes=[w_const.ERROR_IO_PENDING])