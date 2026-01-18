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
@ddt.data((1, 2, 1, 2), (1, 2, 1, 3), (1, 2, 2, 2))
@ddt.unpack
@mock.patch.object(pathutils.PathUtils, 'get_file_id')
def test_is_same_file(self, volume_id_a, file_id_a, volume_id_b, file_id_b, mock_get_file_id):
    file_info_a = self._get_file_id_info(volume_id_a, file_id_a, as_dict=True)
    file_info_b = self._get_file_id_info(volume_id_b, file_id_b, as_dict=True)
    mock_get_file_id.side_effect = [file_info_a, file_info_b]
    same_file = self._pathutils.is_same_file(mock.sentinel.path_a, mock.sentinel.path_b)
    self.assertEqual(volume_id_a == volume_id_b and file_id_a == file_id_b, same_file)
    mock_get_file_id.assert_has_calls([mock.call(mock.sentinel.path_a), mock.call(mock.sentinel.path_b)])