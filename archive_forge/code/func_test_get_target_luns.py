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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_target_sessions')
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_session_disk_luns')
def test_get_target_luns(self, mock_get_iscsi_session_disk_luns, mock_get_iscsi_target_sessions):
    fake_session = mock.Mock()
    mock_get_iscsi_target_sessions.return_value = [fake_session]
    retrieved_luns = [mock.sentinel.lun_0]
    mock_get_iscsi_session_disk_luns.return_value = retrieved_luns
    resulted_luns = self._initiator.get_target_luns(mock.sentinel.target_name)
    mock_get_iscsi_target_sessions.assert_called_once_with(mock.sentinel.target_name)
    mock_get_iscsi_session_disk_luns.assert_called_once_with(fake_session.SessionId)
    self.assertEqual(retrieved_luns, resulted_luns)