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
def test_get_event_listener_stopped(self):
    self._listener._running = False
    self.assertRaises(exceptions.OSWinException, self._listener.get, timeout=1)

    def fake_get(block=True, timeout=0):
        self._listener._running = False
        return None
    self._listener._running = True
    self._listener._event_queue = mock.Mock(get=fake_get)
    self.assertRaises(exceptions.OSWinException, self._listener.get, timeout=1)