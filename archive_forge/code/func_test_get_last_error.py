from unittest import mock
import ddt
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
def test_get_last_error(self):
    last_err = self._win32_utils.get_last_error()
    self.assertEqual(win32utils.kernel32.GetLastError.return_value, last_err)
    win32utils.kernel32.SetLastError.assert_called_once_with(0)