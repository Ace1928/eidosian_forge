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
@mock.patch.object(pathutils.PathUtils, 'rmtree')
@mock.patch.object(pathutils.PathUtils, 'exists')
def test_check_remove_dir(self, mock_exists, mock_rmtree):
    fake_dir = 'dir'
    self._pathutils.check_remove_dir(fake_dir)
    mock_exists.assert_called_once_with(fake_dir)
    mock_rmtree.assert_called_once_with(fake_dir)