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
@mock.patch.object(clusterutils._ClusterEventListener, '_setup')
@mock.patch.object(clusterutils.time, 'sleep')
def test_listen_ignore_exception(self, mock_sleep, mock_setup):
    self._setup_listener(stop_on_error=False)
    self._clusapi.get_cluster_notify_v2.side_effect = (test_base.TestingException, KeyboardInterrupt)
    self.assertRaises(KeyboardInterrupt, self._listener._listen)
    self.assertTrue(self._listener._running)
    mock_sleep.assert_called_once_with(self._listener._error_sleep_interval)