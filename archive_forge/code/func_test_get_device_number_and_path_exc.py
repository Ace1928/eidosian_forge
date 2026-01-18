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
@ddt.data(True, False)
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'ensure_lun_available')
def test_get_device_number_and_path_exc(self, fail_if_not_found, mock_ensure_lun_available):
    raised_exc = exceptions.ISCSILunNotAvailable
    mock_ensure_lun_available.side_effect = raised_exc(target_iqn=mock.sentinel.target_iqn, target_lun=mock.sentinel.target_lun)
    if fail_if_not_found:
        self.assertRaises(raised_exc, self._initiator.get_device_number_and_path, mock.sentinel.target_name, mock.sentinel.lun, fail_if_not_found)
    else:
        dev_num, dev_path = self._initiator.get_device_number_and_path(mock.sentinel.target_name, mock.sentinel.lun, fail_if_not_found)
        self.assertIsNone(dev_num)
        self.assertIsNone(dev_path)