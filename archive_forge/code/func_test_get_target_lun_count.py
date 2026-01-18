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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, 'get_target_luns')
def test_get_target_lun_count(self, mock_get_target_luns):
    target_luns = [mock.sentinel.lun0, mock.sentinel.lun1]
    mock_get_target_luns.return_value = target_luns
    lun_count = self._initiator.get_target_lun_count(mock.sentinel.target_name)
    self.assertEqual(len(target_luns), lun_count)
    mock_get_target_luns.assert_called_once_with(mock.sentinel.target_name)