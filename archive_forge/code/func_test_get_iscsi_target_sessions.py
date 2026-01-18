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
@mock.patch.object(iscsi_utils.ISCSIInitiatorUtils, '_get_iscsi_sessions')
def test_get_iscsi_target_sessions(self, mock_get_iscsi_sessions, target_sessions_found=True):
    fake_session = mock.Mock(TargetNodeName='FAKE_TARGET_NAME', ConnectionCount=1)
    fake_disconn_session = mock.Mock(TargetNodeName='fake_target_name', ConnectionCount=0)
    other_session = mock.Mock(TargetNodeName='other_target_name', ConnectionCount=1)
    sessions = [fake_session, fake_disconn_session, other_session]
    mock_get_iscsi_sessions.return_value = sessions
    resulted_tgt_sessions = self._initiator._get_iscsi_target_sessions('fake_target_name')
    self.assertEqual([fake_session], resulted_tgt_sessions)