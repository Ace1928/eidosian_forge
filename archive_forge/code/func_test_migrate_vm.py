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
@mock.patch.object(wintypes, 'DWORD')
@mock.patch.object(clusterutils.ClusterUtils, '_wait_for_cluster_group_migration')
@mock.patch.object(clusterutils.ClusterUtils, '_validate_migration')
@mock.patch.object(clusterutils, '_ClusterGroupStateChangeListener')
@ddt.data(None, exceptions.ClusterException)
def test_migrate_vm(self, wait_unexpected_exc, mock_listener_cls, mock_validate_migr, mock_wait_group, mock_dword):
    mock_wait_group.side_effect = wait_unexpected_exc
    migrate_args = (self._FAKE_VM_NAME, self._FAKE_HOST, self._clusterutils._LIVE_MIGRATION_TYPE, constants.CLUSTER_GROUP_ONLINE, mock.sentinel.timeout)
    if wait_unexpected_exc:
        self.assertRaises(wait_unexpected_exc, self._clusterutils._migrate_vm, *migrate_args)
    else:
        self._clusterutils._migrate_vm(*migrate_args)
    mock_dword.assert_called_once_with(self._clusterutils._LIVE_MIGRATION_TYPE)
    self._clusapi.get_property_list_entry.assert_has_calls([mock.call(prop_name, w_const.CLUSPROP_SYNTAX_LIST_VALUE_DWORD, mock_dword.return_value) for prop_name in (w_const.CLUS_RESTYPE_NAME_VM, w_const.CLUS_RESTYPE_NAME_VM_CONFIG)])
    expected_prop_entries = [self._clusapi.get_property_list_entry.return_value] * 2
    self._clusapi.get_property_list.assert_called_once_with(expected_prop_entries)
    expected_migrate_flags = w_const.CLUSAPI_GROUP_MOVE_RETURN_TO_SOURCE_NODE_ON_ERROR | w_const.CLUSAPI_GROUP_MOVE_QUEUE_ENABLED | w_const.CLUSAPI_GROUP_MOVE_HIGH_PRIORITY_START
    exp_clus_h = self._cmgr_val(self._cmgr.open_cluster)
    exp_clus_node_h = self._cmgr_val(self._cmgr.open_cluster_node)
    exp_clus_group_h = self._cmgr_val(self._cmgr.open_cluster_group)
    self._cmgr.open_cluster.assert_called_once_with()
    self._cmgr.open_cluster_group.assert_called_once_with(self._FAKE_VM_NAME, cluster_handle=exp_clus_h)
    self._cmgr.open_cluster_node.assert_called_once_with(self._FAKE_HOST, cluster_handle=exp_clus_h)
    self._clusapi.move_cluster_group.assert_called_once_with(exp_clus_group_h, exp_clus_node_h, expected_migrate_flags, self._clusapi.get_property_list.return_value)
    mock_listener_cls.assert_called_once_with(exp_clus_h, self._FAKE_VM_NAME)
    mock_listener = mock_listener_cls.return_value
    mock_wait_group.assert_called_once_with(mock_listener.__enter__.return_value, self._FAKE_VM_NAME, exp_clus_group_h, constants.CLUSTER_GROUP_ONLINE, mock.sentinel.timeout)
    if not wait_unexpected_exc:
        mock_validate_migr.assert_called_once_with(exp_clus_group_h, self._FAKE_VM_NAME, constants.CLUSTER_GROUP_ONLINE, self._FAKE_HOST)