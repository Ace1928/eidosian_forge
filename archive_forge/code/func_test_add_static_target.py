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
def test_add_static_target(self):
    self._mock_ctypes()
    is_persistent = True
    self._initiator._add_static_target(mock.sentinel.target_name, is_persistent=is_persistent)
    self._mock_run.assert_called_once_with(self._iscsidsc.AddIScsiStaticTargetW, self._ctypes.c_wchar_p(mock.sentinel.target_name), None, 0, is_persistent, None, None, None)