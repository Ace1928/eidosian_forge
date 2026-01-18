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
@ddt.data(0, w_const.ERROR_IO_PENDING)
def test_cancel_cluster_group_operation(self, cancel_ret_val):
    self._mock_run.return_value = cancel_ret_val
    expected_ret_val = cancel_ret_val != w_const.ERROR_IO_PENDING
    ret_val = self._clusapi_utils.cancel_cluster_group_operation(mock.sentinel.group_handle)
    self.assertEqual(expected_ret_val, ret_val)
    self._mock_run.assert_called_once_with(self._clusapi.CancelClusterGroupOperation, mock.sentinel.group_handle, 0, ignored_error_codes=[w_const.ERROR_IO_PENDING])