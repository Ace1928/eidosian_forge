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
@mock.patch('os.path.isdir')
def test_copy_exc(self, mock_isdir):
    mock_isdir.return_value = False
    self._mock_run.side_effect = exceptions.Win32Exception(func_name='mock_copy', error_code='fake_error_code', error_message='fake_error_msg')
    self.assertRaises(IOError, self._pathutils.copy, mock.sentinel.src, mock.sentinel.dest)