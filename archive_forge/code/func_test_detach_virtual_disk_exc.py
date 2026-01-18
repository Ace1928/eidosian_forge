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
@ddt.data(True, False)
@mock.patch('os.path.exists')
@mock.patch.object(vhdutils.VHDUtils, '_open')
@mock.patch.object(vhdutils.VHDUtils, 'is_virtual_disk_file_attached')
def test_detach_virtual_disk_exc(self, is_attached, mock_is_attached, mock_open, mock_exists):
    mock_exists.return_value = True
    mock_is_attached.return_value = is_attached
    mock_open.side_effect = exceptions.Win32Exception(message='fake exc')
    if is_attached:
        self.assertRaises(exceptions.Win32Exception, self._vhdutils.detach_virtual_disk, mock.sentinel.vhd_path)
    else:
        self._vhdutils.detach_virtual_disk(mock.sentinel.vhd_path)
    mock_is_attached.assert_called_once_with(mock.sentinel.vhd_path)