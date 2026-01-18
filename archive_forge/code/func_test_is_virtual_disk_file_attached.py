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
@ddt.data({}, {'exists': False}, {'open_fails': True})
@ddt.unpack
@mock.patch('os.path.exists')
@mock.patch.object(vhdutils.VHDUtils, 'get_vhd_info')
def test_is_virtual_disk_file_attached(self, mock_get_vhd_info, mock_exists, exists=True, open_fails=False):
    mock_exists.return_value = exists
    if open_fails:
        mock_get_vhd_info.side_effect = exceptions.Win32Exception(message='fake exc')
    else:
        mock_get_vhd_info.return_value = {'IsLoaded': mock.sentinel.attached}
    fallback = self._disk_utils.is_virtual_disk_file_attached
    fallback.return_value = True
    ret_val = self._vhdutils.is_virtual_disk_file_attached(mock.sentinel.vhd_path)
    exp_ret_val = True if exists else False
    self.assertEqual(exp_ret_val, ret_val)
    if exists:
        mock_get_vhd_info.assert_called_once_with(mock.sentinel.vhd_path, [w_const.GET_VIRTUAL_DISK_INFO_IS_LOADED])
    else:
        mock_get_vhd_info.assert_not_called()
    if exists and open_fails:
        fallback.assert_called_once_with(mock.sentinel.vhd_path)
    else:
        fallback.assert_not_called()