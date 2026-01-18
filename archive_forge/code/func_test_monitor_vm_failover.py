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
@mock.patch.object(clusterutils, 'tpool')
@mock.patch.object(clusterutils, 'patcher')
def test_monitor_vm_failover(self, mock_patcher, mock_tpool):
    mock_watcher = mock.MagicMock()
    fake_prev = mock.MagicMock(OwnerNode=self._FAKE_PREV_HOST)
    fake_wmi_object = mock.MagicMock(OwnerNode=self._FAKE_HOST, Name=self._FAKE_RESOURCEGROUP_NAME, previous=fake_prev)
    mock_tpool.execute.return_value = fake_wmi_object
    fake_callback = mock.MagicMock()
    self._clusterutils._monitor_vm_failover(mock_watcher, fake_callback)
    mock_tpool.execute.assert_called_once_with(mock_watcher, self._clusterutils._WMI_EVENT_TIMEOUT_MS)
    fake_callback.assert_called_once_with(self._FAKE_VM_NAME, self._FAKE_PREV_HOST, self._FAKE_HOST)