import collections
import ctypes
from unittest import mock
import ddt
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import iscsi_utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def test_get_items_from_buff(self):
    fake_buff_contents = 'fake_buff_contents'
    fake_buff = (ctypes.c_wchar * len(fake_buff_contents))()
    fake_buff.value = fake_buff_contents
    fake_buff = ctypes.cast(fake_buff, ctypes.POINTER(ctypes.c_ubyte))
    result = iscsi_utils._get_items_from_buff(fake_buff, ctypes.c_wchar, len(fake_buff_contents))
    self.assertEqual(fake_buff_contents, result.value)