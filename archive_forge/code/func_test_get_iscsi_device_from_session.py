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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_session_devices')
def test_get_iscsi_device_from_session(self, mock_get_iscsi_session_devices):
    fake_device = mock.Mock()
    fake_device.ScsiAddress.Lun = mock.sentinel.target_lun
    mock_get_iscsi_session_devices.return_value = [mock.Mock(), fake_device]
    resulted_device = self._initiator._get_iscsi_device_from_session(mock.sentinel.session_id, mock.sentinel.target_lun)
    mock_get_iscsi_session_devices.assert_called_once_with(mock.sentinel.session_id)
    self.assertEqual(fake_device, resulted_device)