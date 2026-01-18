import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
@mock.patch('ctypes.byref')
def test_get_cluster_group_type(self, mock_byref):
    mock_byref.side_effect = lambda x: ('byref', x)
    self._clusapi.cluster_group_control.return_value = (mock.sentinel.buff, mock.sentinel.buff_sz)
    ret_val = self._clusterutils.get_cluster_group_type(mock.sentinel.group_name)
    self.assertEqual(self._clusapi.get_cluster_group_type.return_value, ret_val)
    self._cmgr.open_cluster_group.assert_called_once_with(mock.sentinel.group_name)
    self._clusapi.cluster_group_control.assert_called_once_with(self._cmgr_val(self._cmgr.open_cluster_group), w_const.CLUSCTL_GROUP_GET_RO_COMMON_PROPERTIES)
    self._clusapi.get_cluster_group_type.assert_called_once_with(mock_byref(mock.sentinel.buff), mock.sentinel.buff_sz)