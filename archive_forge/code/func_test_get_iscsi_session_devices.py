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
def test_get_iscsi_session_devices(self):
    self._mock_ctypes()
    _get_iscsi_session_devices = _utils.get_wrapped_function(self._initiator._get_iscsi_session_devices)
    _get_iscsi_session_devices(self._initiator, mock.sentinel.session_id, buff=mock.sentinel.buff, element_count=mock.sentinel.element_count)
    self._mock_run.assert_called_once_with(self._iscsidsc.GetDevicesForIScsiSessionW, self._ctypes.byref(mock.sentinel.session_id), self._ctypes.byref(mock.sentinel.element_count), mock.sentinel.buff)