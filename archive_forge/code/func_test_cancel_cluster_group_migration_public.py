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
@mock.patch.object(clusterutils.ClusterUtils, '_cancel_cluster_group_migration')
@mock.patch.object(clusterutils, '_ClusterGroupStateChangeListener')
def test_cancel_cluster_group_migration_public(self, mock_listener_cls, mock_cancel_migr):
    exp_clus_h = self._cmgr_val(self._cmgr.open_cluster)
    exp_clus_group_h = self._cmgr_val(self._cmgr.open_cluster_group)
    mock_listener = mock_listener_cls.return_value
    mock_listener.__enter__.return_value = mock_listener
    self._clusterutils.cancel_cluster_group_migration(mock.sentinel.group_name, mock.sentinel.expected_state, mock.sentinel.timeout)
    self._cmgr.open_cluster.assert_called_once_with()
    self._cmgr.open_cluster_group.assert_called_once_with(mock.sentinel.group_name, cluster_handle=exp_clus_h)
    mock_listener.__enter__.assert_called_once_with()
    mock_listener_cls.assert_called_once_with(exp_clus_h, mock.sentinel.group_name)
    mock_cancel_migr.assert_called_once_with(mock_listener, mock.sentinel.group_name, exp_clus_group_h, mock.sentinel.expected_state, mock.sentinel.timeout)