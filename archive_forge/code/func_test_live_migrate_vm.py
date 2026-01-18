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
@mock.patch.object(clusterutils.ClusterUtils, '_migrate_vm')
def test_live_migrate_vm(self, mock_migrate_vm):
    self._clusterutils.live_migrate_vm(self._FAKE_VM_NAME, self._FAKE_HOST, mock.sentinel.timeout)
    mock_migrate_vm.assert_called_once_with(self._FAKE_VM_NAME, self._FAKE_HOST, self._clusterutils._LIVE_MIGRATION_TYPE, constants.CLUSTER_GROUP_ONLINE, mock.sentinel.timeout)