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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'ensure_lun_available')
def test_get_device_number_and_path(self, mock_ensure_lun_available):
    mock_ensure_lun_available.return_value = (mock.sentinel.dev_num, mock.sentinel.dev_path)
    dev_num, dev_path = self._initiator.get_device_number_and_path(mock.sentinel.target_name, mock.sentinel.lun, retry_attempts=mock.sentinel.retry_attempts, retry_interval=mock.sentinel.retry_interval, rescan_disks=mock.sentinel.rescan_disks, ensure_mpio_claimed=mock.sentinel.ensure_mpio_claimed)
    mock_ensure_lun_available.assert_called_once_with(mock.sentinel.target_name, mock.sentinel.lun, rescan_attempts=mock.sentinel.retry_attempts, retry_interval=mock.sentinel.retry_interval, rescan_disks=mock.sentinel.rescan_disks, ensure_mpio_claimed=mock.sentinel.ensure_mpio_claimed)
    self.assertEqual(mock.sentinel.dev_num, dev_num)
    self.assertEqual(mock.sentinel.dev_path, dev_path)