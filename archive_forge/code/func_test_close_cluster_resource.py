import ctypes
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import _clusapi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def test_close_cluster_resource(self):
    self._clusapi_utils.close_cluster_resource(mock.sentinel.handle)
    self._clusapi.CloseClusterResource.assert_called_once_with(mock.sentinel.handle)