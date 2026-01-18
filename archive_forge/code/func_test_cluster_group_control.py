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
def test_cluster_group_control(self):
    fake_out_buff = 'fake-event-buff'
    requested_buff_sz = 1024

    def fake_cluster_group_ctrl(func, group_handle, node_handle, control_code, in_buff_p, in_buff_sz, out_buff_p, out_buff_sz, requested_buff_sz_p):
        self.assertEqual(self._clusapi.ClusterGroupControl, func)
        self.assertEqual(mock.sentinel.group_handle, group_handle)
        self.assertEqual(mock.sentinel.node_handle, node_handle)
        self.assertEqual(mock.sentinel.control_code, control_code)
        self.assertEqual(mock.sentinel.in_buff_p, in_buff_p)
        self.assertEqual(mock.sentinel.in_buff_sz, in_buff_sz)
        req_buff_sz = ctypes.cast(requested_buff_sz_p, wintypes.PDWORD).contents
        req_buff_sz.value = requested_buff_sz
        if out_buff_sz.value < requested_buff_sz:
            raise exceptions.ClusterWin32Exception(error_code=w_const.ERROR_MORE_DATA, func_name='ClusterGroupControl', error_message='error more data')
        out_buff = ctypes.cast(out_buff_p, ctypes.POINTER(ctypes.c_wchar * (requested_buff_sz // ctypes.sizeof(ctypes.c_wchar))))
        out_buff = out_buff.contents
        out_buff.value = fake_out_buff
    self._mock_run.side_effect = fake_cluster_group_ctrl
    out_buff, out_buff_sz = self._clusapi_utils.cluster_group_control(mock.sentinel.group_handle, mock.sentinel.control_code, mock.sentinel.node_handle, mock.sentinel.in_buff_p, mock.sentinel.in_buff_sz)
    self.assertEqual(requested_buff_sz, out_buff_sz)
    wp_out_buff = ctypes.cast(out_buff, ctypes.POINTER(ctypes.c_wchar * requested_buff_sz))
    self.assertEqual(fake_out_buff, wp_out_buff.contents[:len(fake_out_buff)])