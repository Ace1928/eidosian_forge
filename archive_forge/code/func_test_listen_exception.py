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
def test_listen_exception(self):
    self._clusapi.get_cluster_notify_v2.side_effect = test_base.TestingException
    self._listener._listen()
    self.assertFalse(self._listener._running)