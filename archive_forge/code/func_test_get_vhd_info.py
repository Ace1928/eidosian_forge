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
@mock.patch.object(vhdutils.VHDUtils, '_open')
@mock.patch.object(vhdutils.VHDUtils, '_get_vhd_info_member')
def test_get_vhd_info(self, mock_get_vhd_info_member, mock_open):
    fake_info_member = w_const.GET_VIRTUAL_DISK_INFO_SIZE
    fake_vhd_info = {'VirtualSize': mock.sentinel.virtual_size}
    mock_open.return_value = mock.sentinel.handle
    mock_get_vhd_info_member.return_value = fake_vhd_info
    expected_open_flag = w_const.OPEN_VIRTUAL_DISK_FLAG_NO_PARENTS
    expected_access_mask = w_const.VIRTUAL_DISK_ACCESS_GET_INFO | w_const.VIRTUAL_DISK_ACCESS_DETACH
    ret_val = self._vhdutils.get_vhd_info(mock.sentinel.vhd_path, [fake_info_member])
    self.assertEqual(fake_vhd_info, ret_val)
    mock_open.assert_called_once_with(mock.sentinel.vhd_path, open_flag=expected_open_flag, open_access_mask=expected_access_mask)
    self._vhdutils._get_vhd_info_member.assert_called_once_with(mock.sentinel.handle, fake_info_member)
    self._mock_close.assert_called_once_with(mock.sentinel.handle)