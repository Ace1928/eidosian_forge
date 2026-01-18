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
def cluster_group_control(self, group_handle, control_code, node_handle=None, in_buff_p=None, in_buff_sz=0):
    out_buff_sz = ctypes.c_ulong(w_const.MAX_PATH)
    out_buff = (ctypes.c_ubyte * out_buff_sz.value)()

    def get_args(out_buff):
        return (clusapi.ClusterGroupControl, group_handle, node_handle, control_code, in_buff_p, in_buff_sz, out_buff, out_buff_sz, ctypes.byref(out_buff_sz))
    try:
        self._run_and_check_output(*get_args(out_buff))
    except exceptions.ClusterWin32Exception as ex:
        if ex.error_code == w_const.ERROR_MORE_DATA:
            out_buff = (ctypes.c_ubyte * out_buff_sz.value)()
            self._run_and_check_output(*get_args(out_buff))
        else:
            raise
    return (out_buff, out_buff_sz.value)