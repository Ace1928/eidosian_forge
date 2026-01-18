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
@mock.patch.object(clusterutils._ClusterEventListener, '_get_notif_key_dw')
def test_add_filter(self, mock_get_notif_key):
    mock_get_notif_key.side_effect = (mock.sentinel.notif_key_dw, mock.sentinel.notif_key_dw_2)
    self._clusapi.create_cluster_notify_port_v2.return_value = mock.sentinel.notif_port_h
    self._listener._add_filter(mock.sentinel.filter, mock.sentinel.notif_key)
    self._listener._add_filter(mock.sentinel.filter_2, mock.sentinel.notif_key_2)
    self.assertEqual(mock.sentinel.notif_port_h, self._listener._notif_port_h)
    mock_get_notif_key.assert_has_calls([mock.call(mock.sentinel.notif_key), mock.call(mock.sentinel.notif_key_2)])
    self._clusapi.create_cluster_notify_port_v2.assert_has_calls([mock.call(mock.sentinel.cluster_handle, mock.sentinel.filter, None, mock.sentinel.notif_key_dw), mock.call(mock.sentinel.cluster_handle, mock.sentinel.filter_2, mock.sentinel.notif_port_h, mock.sentinel.notif_key_dw_2)])