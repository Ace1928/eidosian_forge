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
@mock.patch.object(clusterutils.ClusterUtils, '_wait_for_cluster_group_migration')
@mock.patch.object(clusterutils.ClusterUtils, '_validate_migration')
@mock.patch.object(clusterutils, '_ClusterGroupStateChangeListener')
@ddt.data(True, False)
def test_migrate_vm_timeout(self, finished_after_cancel, mock_listener_cls, mock_validate_migr, mock_wait_group, mock_cancel_migr):
    timeout_exc = exceptions.ClusterGroupMigrationTimeOut(group_name=self._FAKE_VM_NAME, time_elapsed=10)
    mock_wait_group.side_effect = timeout_exc
    mock_listener = self._cmgr_val(mock_listener_cls)
    mock_validate_migr.side_effect = (None,) if finished_after_cancel else exceptions.ClusterGroupMigrationFailed(group_name=self._FAKE_VM_NAME, expected_state=mock.sentinel.expected_state, expected_node=self._FAKE_HOST, group_state=mock.sentinel.expected_state, owner_node=mock.sentinel.other_host)
    migrate_args = (self._FAKE_VM_NAME, self._FAKE_HOST, self._clusterutils._LIVE_MIGRATION_TYPE, mock.sentinel.exp_state, mock.sentinel.timeout)
    if finished_after_cancel:
        self._clusterutils._migrate_vm(*migrate_args)
    else:
        self.assertRaises(exceptions.ClusterGroupMigrationTimeOut, self._clusterutils._migrate_vm, *migrate_args)
    exp_clus_group_h = self._cmgr_val(self._cmgr.open_cluster_group)
    mock_cancel_migr.assert_called_once_with(mock_listener, self._FAKE_VM_NAME, exp_clus_group_h, mock.sentinel.exp_state, mock.sentinel.timeout)
    mock_validate_migr.assert_called_once_with(exp_clus_group_h, self._FAKE_VM_NAME, mock.sentinel.exp_state, self._FAKE_HOST)