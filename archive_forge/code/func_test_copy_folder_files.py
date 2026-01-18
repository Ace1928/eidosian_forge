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
@mock.patch.object(pathutils.PathUtils, 'copy')
@mock.patch.object(os.path, 'isfile')
@mock.patch.object(os, 'listdir')
@mock.patch.object(pathutils.PathUtils, 'check_create_dir')
def test_copy_folder_files(self, mock_check_create_dir, mock_listdir, mock_isfile, mock_copy):
    src_dir = 'src'
    dest_dir = 'dest'
    fname = 'tmp_file.txt'
    subdir = 'tmp_folder'
    src_fname = os.path.join(src_dir, fname)
    dest_fname = os.path.join(dest_dir, fname)
    mock_listdir.return_value = [fname, subdir]
    mock_isfile.side_effect = [True, False]
    self._pathutils.copy_folder_files(src_dir, dest_dir)
    mock_check_create_dir.assert_called_once_with(dest_dir)
    mock_copy.assert_called_once_with(src_fname, dest_fname)