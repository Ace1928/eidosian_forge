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
def test_cluster_get_enum_count(self):
    ret_val = self._clusapi_utils.cluster_get_enum_count(mock.sentinel.enum_handle)
    self.assertEqual(self._mock_run.return_value, ret_val)
    self._mock_run.assert_called_once_with(self._clusapi.ClusterGetEnumCountEx, mock.sentinel.enum_handle, error_on_nonzero_ret_val=False, ret_val_is_err_code=False)