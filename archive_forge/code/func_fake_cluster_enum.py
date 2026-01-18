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
def fake_cluster_enum(func, enum_handle, index, buff_p, buff_sz_p, ignored_error_codes=tuple()):
    self.assertEqual(self._clusapi.ClusterEnumEx, func)
    self.assertEqual(mock.sentinel.enum_handle, enum_handle)
    self.assertEqual(mock.sentinel.index, index)
    buff_sz = ctypes.cast(buff_sz_p, wintypes.PDWORD).contents
    if buff_sz.value < requested_buff_sz:
        buff_sz.value = requested_buff_sz
        if w_const.ERROR_MORE_DATA not in ignored_error_codes:
            raise exceptions.ClusterWin32Exception(error_code=w_const.ERROR_MORE_DATA)
        return
    item = ctypes.cast(buff_p, clusapi_def.PCLUSTER_ENUM_ITEM).contents
    item.lpszId = obj_id_wchar_p
    item.cbId = len(obj_id)