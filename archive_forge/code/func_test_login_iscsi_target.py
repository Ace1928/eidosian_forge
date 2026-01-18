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
@mock.patch.object(ctypes, 'byref')
@mock.patch.object(iscsi_struct, 'ISCSI_UNIQUE_CONNECTION_ID')
@mock.patch.object(iscsi_struct, 'ISCSI_UNIQUE_SESSION_ID')
def test_login_iscsi_target(self, mock_cls_ISCSI_UNIQUE_SESSION_ID, mock_cls_ISCSI_UNIQUE_CONNECTION_ID, mock_byref):
    fake_target_name = 'fake_target_name'
    resulted_session_id, resulted_conection_id = self._initiator._login_iscsi_target(fake_target_name)
    args_list = self._mock_run.call_args_list[0][0]
    self.assertIsInstance(args_list[1], ctypes.c_wchar_p)
    self.assertEqual(fake_target_name, args_list[1].value)
    self.assertIsInstance(args_list[4], ctypes.c_ulong)
    self.assertEqual(ctypes.c_ulong(w_const.ISCSI_ANY_INITIATOR_PORT).value, args_list[4].value)
    self.assertIsInstance(args_list[6], ctypes.c_ulonglong)
    self.assertEqual(0, args_list[6].value)
    self.assertIsInstance(args_list[9], ctypes.c_ulong)
    self.assertEqual(0, args_list[9].value)
    mock_byref.assert_has_calls([mock.call(mock_cls_ISCSI_UNIQUE_SESSION_ID.return_value), mock.call(mock_cls_ISCSI_UNIQUE_CONNECTION_ID.return_value)])
    self.assertEqual(mock_cls_ISCSI_UNIQUE_SESSION_ID.return_value, resulted_session_id)
    self.assertEqual(mock_cls_ISCSI_UNIQUE_CONNECTION_ID.return_value, resulted_conection_id)