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
def test_get_file_id_exc(self):
    self._mock_run.side_effect = exceptions.Win32Exception(message='fake exc')
    self.assertRaises(exceptions.Win32Exception, self._pathutils.get_file_id, mock.sentinel.path)
    self._io_utils.close_handle.assert_called_once_with(self._io_utils.open.return_value)