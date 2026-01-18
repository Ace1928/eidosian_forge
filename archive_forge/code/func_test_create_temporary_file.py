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
@mock.patch('os.close')
@mock.patch('tempfile.mkstemp')
def test_create_temporary_file(self, mock_mkstemp, mock_close):
    fd = mock.sentinel.file_descriptor
    path = mock.sentinel.absolute_pathname
    mock_mkstemp.return_value = (fd, path)
    output = self._pathutils.create_temporary_file(suffix=mock.sentinel.suffix)
    self.assertEqual(path, output)
    mock_close.assert_called_once_with(fd)
    mock_mkstemp.assert_called_once_with(suffix=mock.sentinel.suffix)