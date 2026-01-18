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
def test_cluster_enum(self):
    cluster_objects = [mock.Mock(), mock.Mock()]
    self._clusapi.cluster_get_enum_count.return_value = len(cluster_objects)
    self._clusapi.cluster_enum.side_effect = cluster_objects
    exp_ret_val = [dict(version=item.dwVersion, type=item.dwType, id=item.lpszId, name=item.lpszName) for item in cluster_objects]
    ret_val = list(self._clusterutils.cluster_enum(mock.sentinel.obj_type))
    self.assertEqual(exp_ret_val, ret_val)
    enum_handle = self._cmgr_val(self._cmgr.open_cluster_enum)
    self._cmgr.open_cluster_enum.assert_called_once_with(mock.sentinel.obj_type)
    self._clusapi.cluster_get_enum_count.assert_called_once_with(enum_handle)
    self._clusapi.cluster_enum.assert_has_calls([mock.call(enum_handle, idx) for idx in range(len(cluster_objects))])