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
def test_get_cluster_group_state(self, mock_byref):
    mock_byref.side_effect = lambda x: ('byref', x)
    state_info = dict(state=mock.sentinel.state, owner_node=mock.sentinel.owner_node)
    self._clusapi.get_cluster_group_state.return_value = state_info
    self._clusapi.cluster_group_control.return_value = (mock.sentinel.buff, mock.sentinel.buff_sz)
    self._clusapi.get_cluster_group_status_info.return_value = mock.sentinel.status_info
    exp_state_info = state_info.copy()
    exp_state_info['status_info'] = mock.sentinel.status_info
    ret_val = self._clusterutils._get_cluster_group_state(mock.sentinel.group_handle)
    self.assertEqual(exp_state_info, ret_val)
    self._clusapi.get_cluster_group_state.assert_called_once_with(mock.sentinel.group_handle)
    self._clusapi.cluster_group_control.assert_called_once_with(mock.sentinel.group_handle, w_const.CLUSCTL_GROUP_GET_RO_COMMON_PROPERTIES)
    self._clusapi.get_cluster_group_status_info.assert_called_once_with(mock_byref(mock.sentinel.buff), mock.sentinel.buff_sz)