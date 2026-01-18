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
def test_ensure_buff_func_unexpected_exception(self):
    fake_exc = exceptions.Win32Exception(message='fake_message', error_code=1)
    func_side_effect = mock.Mock(side_effect=fake_exc)
    fake_func = self._get_fake_iscsi_utils_getter_func(func_side_effect=func_side_effect, decorator_args={'struct_type': ctypes.c_ubyte})
    self.assertRaises(exceptions.Win32Exception, fake_func, self._initiator)