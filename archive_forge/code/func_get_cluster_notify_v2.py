import contextlib
import ctypes
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
def get_cluster_notify_v2(self, notif_port_h, timeout_ms):
    filter_and_type = clusapi_def.NOTIFY_FILTER_AND_TYPE()
    obj_name_buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    obj_type_buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    obj_id_buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    parent_id_buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    notif_key_p = wintypes.PDWORD()
    buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    buff = (wintypes.BYTE * buff_sz.value)()
    obj_name_buff = (ctypes.c_wchar * obj_name_buff_sz.value)()
    obj_type_buff = (ctypes.c_wchar * obj_type_buff_sz.value)()
    obj_id_buff = (ctypes.c_wchar * obj_id_buff_sz.value)()
    parent_id_buff = (ctypes.c_wchar * parent_id_buff_sz.value)()
    try:
        self._run_and_check_output(clusapi.GetClusterNotifyV2, notif_port_h, ctypes.byref(notif_key_p), ctypes.byref(filter_and_type), buff, ctypes.byref(buff_sz), obj_id_buff, ctypes.byref(obj_id_buff_sz), parent_id_buff, ctypes.byref(parent_id_buff_sz), obj_name_buff, ctypes.byref(obj_name_buff_sz), obj_type_buff, ctypes.byref(obj_type_buff_sz), timeout_ms)
    except exceptions.ClusterWin32Exception as ex:
        if ex.error_code == w_const.ERROR_MORE_DATA:
            buff = (wintypes.BYTE * buff_sz.value)()
            obj_name_buff = (ctypes.c_wchar * obj_name_buff_sz.value)()
            parent_id_buff = (ctypes.c_wchar * parent_id_buff_sz.value)()
            obj_type_buff = (ctypes.c_wchar * obj_type_buff_sz.value)()
            obj_id_buff = (ctypes.c_wchar * obj_id_buff_sz.value)()
            self._run_and_check_output(clusapi.GetClusterNotifyV2, notif_port_h, ctypes.byref(notif_key_p), ctypes.byref(filter_and_type), buff, ctypes.byref(buff_sz), obj_id_buff, ctypes.byref(obj_id_buff_sz), parent_id_buff, ctypes.byref(parent_id_buff_sz), obj_name_buff, ctypes.byref(obj_name_buff_sz), obj_type_buff, ctypes.byref(obj_type_buff_sz), timeout_ms)
        else:
            raise
    notif_key = notif_key_p.contents.value
    event = {'cluster_object_name': obj_name_buff.value, 'object_id': obj_id_buff.value, 'object_type': filter_and_type.dwObjectType, 'object_type_str': obj_type_buff.value, 'filter_flags': filter_and_type.FilterFlags, 'parent_id': parent_id_buff.value, 'buff': buff, 'buff_sz': buff_sz.value, 'notif_key': notif_key}
    return event