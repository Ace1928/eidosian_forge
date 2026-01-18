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
def test_get_node_name(self):
    self._clusterutils._this_node = mock.sentinel.fake_node_name
    self.assertEqual(mock.sentinel.fake_node_name, self._clusterutils.get_node_name())