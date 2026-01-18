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
@mock.patch.object(clusterutils.ClusterUtils, '_get_failover_watcher')
@mock.patch.object(clusterutils.ClusterUtils, '_monitor_vm_failover')
@mock.patch.object(clusterutils, 'time')
def test_get_vm_owner_change_listener(self, mock_time, mock_monitor, mock_get_watcher):
    mock_monitor.side_effect = [None, exceptions.OSWinException, KeyboardInterrupt]
    listener = self._clusterutils.get_vm_owner_change_listener()
    self.assertRaises(KeyboardInterrupt, listener, mock.sentinel.callback)
    mock_monitor.assert_has_calls([mock.call(mock_get_watcher.return_value, mock.sentinel.callback, constants.DEFAULT_WMI_EVENT_TIMEOUT_MS)] * 3)
    mock_time.sleep.assert_called_once_with(constants.DEFAULT_WMI_EVENT_TIMEOUT_MS / 1000)