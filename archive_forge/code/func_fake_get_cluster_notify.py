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
def fake_get_cluster_notify(func, notif_port_h, pp_notif_key, p_filter_and_type, p_buff, p_buff_sz, p_obj_id_buff, p_obj_id_buff_sz, p_parent_id_buff, p_parent_id_buff_sz, p_obj_name_buff, p_obj_name_buff_sz, p_obj_type, p_obj_type_sz, timeout_ms):
    self.assertEqual(self._clusapi.GetClusterNotifyV2, func)
    self.assertEqual(fake_notif_port_h, notif_port_h)
    obj_name_buff_sz = ctypes.cast(p_obj_name_buff_sz, wintypes.PDWORD).contents
    buff_sz = ctypes.cast(p_buff_sz, wintypes.PDWORD).contents
    obj_type_sz = ctypes.cast(p_obj_type_sz, wintypes.PDWORD).contents
    obj_id_sz = ctypes.cast(p_obj_id_buff_sz, wintypes.PDWORD).contents
    parent_id_buff_sz = ctypes.cast(p_parent_id_buff_sz, wintypes.PDWORD).contents
    if buff_sz.value < requested_buff_sz or obj_name_buff_sz.value < requested_buff_sz or parent_id_buff_sz.value < requested_buff_sz or (obj_type_sz.value < requested_buff_sz) or (obj_id_sz.value < requested_buff_sz):
        buff_sz.value = requested_buff_sz
        obj_name_buff_sz.value = requested_buff_sz
        parent_id_buff_sz.value = requested_buff_sz
        obj_type_sz.value = requested_buff_sz
        obj_id_sz.value = requested_buff_sz
        raise exceptions.ClusterWin32Exception(error_code=w_const.ERROR_MORE_DATA, func_name='GetClusterNotifyV2', error_message='error more data')
    pp_notif_key = ctypes.cast(pp_notif_key, ctypes.c_void_p)
    p_notif_key = ctypes.c_void_p.from_address(pp_notif_key.value)
    p_notif_key.value = ctypes.addressof(notif_key)
    filter_and_type = ctypes.cast(p_filter_and_type, ctypes.POINTER(clusapi_def.NOTIFY_FILTER_AND_TYPE)).contents
    filter_and_type.dwObjectType = fake_notif_type
    filter_and_type.FilterFlags = fake_filter_flags

    def set_wchar_buff(p_wchar_buff, wchar_buff_sz, value):
        wchar_buff = ctypes.cast(p_wchar_buff, ctypes.POINTER(ctypes.c_wchar * (wchar_buff_sz // ctypes.sizeof(ctypes.c_wchar))))
        wchar_buff = wchar_buff.contents
        ctypes.memset(wchar_buff, 0, wchar_buff_sz)
        wchar_buff.value = value
        return wchar_buff
    set_wchar_buff(p_obj_name_buff, requested_buff_sz, fake_clus_obj_name)
    set_wchar_buff(p_buff, requested_buff_sz, fake_event_buff)
    set_wchar_buff(p_parent_id_buff, requested_buff_sz, fake_parent_id)
    set_wchar_buff(p_obj_type, requested_buff_sz, fake_obj_type)
    set_wchar_buff(p_obj_id_buff, requested_buff_sz, fake_obj_id)
    self.assertEqual(mock.sentinel.timeout_ms, timeout_ms)