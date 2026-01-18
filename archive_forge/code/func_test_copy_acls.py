import ctypes
import os
import shutil
from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import pathutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import advapi32 as advapi32_def
from os_win.utils.winapi.libs import kernel32 as kernel32_def
from os_win.utils.winapi import wintypes
def test_copy_acls(self):
    raised_exc = exceptions.OSWinException
    mock_sec_info = dict(pp_sec_desc=mock.Mock(), pp_dacl=mock.Mock())
    self._acl_utils.get_named_security_info.return_value = mock_sec_info
    self._acl_utils.set_named_security_info.side_effect = raised_exc
    self.assertRaises(raised_exc, self._pathutils.copy_acls, mock.sentinel.src, mock.sentinel.dest)
    self._acl_utils.get_named_security_info.assert_called_once_with(obj_name=mock.sentinel.src, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION)
    self._acl_utils.set_named_security_info.assert_called_once_with(obj_name=mock.sentinel.dest, obj_type=w_const.SE_FILE_OBJECT, security_info_flags=w_const.DACL_SECURITY_INFORMATION, p_dacl=mock_sec_info['pp_dacl'].contents)
    self._pathutils._win32_utils.local_free.assert_called_once_with(mock_sec_info['pp_sec_desc'].contents)