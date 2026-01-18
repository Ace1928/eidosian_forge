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
def test_init_hyperv_conn_exception(self):
    self._clusterutils._get_wmi_conn = mock.MagicMock()
    self._clusterutils._get_wmi_conn.side_effect = AttributeError
    self.assertRaises(exceptions.HyperVClusterException, self._clusterutils._init_hyperv_conn, 'fake_host', timeout=1)