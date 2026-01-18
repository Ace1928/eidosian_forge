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
def set_wchar_buff(p_wchar_buff, wchar_buff_sz, value):
    wchar_buff = ctypes.cast(p_wchar_buff, ctypes.POINTER(ctypes.c_wchar * (wchar_buff_sz // ctypes.sizeof(ctypes.c_wchar))))
    wchar_buff = wchar_buff.contents
    ctypes.memset(wchar_buff, 0, wchar_buff_sz)
    wchar_buff.value = value
    return wchar_buff