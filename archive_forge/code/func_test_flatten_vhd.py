import ctypes
import os
from unittest import mock
import uuid
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.virtdisk import vhdutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import wintypes
@mock.patch.object(vhdutils.VHDUtils, 'convert_vhd')
@mock.patch.object(os, 'unlink')
@mock.patch.object(os, 'rename')
def test_flatten_vhd(self, mock_rename, mock_unlink, mock_convert):
    fake_vhd_path = 'C:\\test.vhd'
    expected_tmp_path = 'C:\\test.tmp.vhd'
    self._vhdutils.flatten_vhd(fake_vhd_path)
    mock_convert.assert_called_once_with(fake_vhd_path, expected_tmp_path)
    mock_unlink.assert_called_once_with(fake_vhd_path)
    mock_rename.assert_called_once_with(expected_tmp_path, fake_vhd_path)