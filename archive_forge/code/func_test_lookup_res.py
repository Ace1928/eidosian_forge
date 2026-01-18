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
def test_lookup_res(self):
    res_list = [mock.sentinel.r1]
    resource_source = mock.MagicMock()
    resource_source.return_value = res_list
    self.assertEqual(mock.sentinel.r1, self._clusterutils._lookup_res(resource_source, self._FAKE_RES_NAME))
    resource_source.assert_called_once_with(Name=self._FAKE_RES_NAME)