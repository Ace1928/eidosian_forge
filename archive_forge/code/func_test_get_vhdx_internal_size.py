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
@mock.patch.object(vhdutils.VHDUtils, '_get_vhdx_block_size')
@mock.patch.object(vhdutils.VHDUtils, '_get_vhdx_log_size')
@mock.patch.object(vhdutils.VHDUtils, '_get_vhdx_metadata_size_and_offset')
def test_get_vhdx_internal_size(self, mock_get_vhdx_md_sz_and_off, mock_get_vhdx_log_sz, mock_get_vhdx_block_size):
    self._mock_open()
    fake_log_sz = 1 << 20
    fake_block_sz = 32 << 20
    fake_md_sz = 1 << 20
    fake_logical_sector_sz = 4096
    new_vhd_sz = 1 << 30
    expected_max_int_sz = new_vhd_sz - fake_block_sz
    fake_vhd_info = dict(SectorSize=fake_logical_sector_sz)
    mock_get_vhdx_block_size.return_value = fake_block_sz
    mock_get_vhdx_log_sz.return_value = fake_log_sz
    mock_get_vhdx_md_sz_and_off.return_value = (fake_md_sz, None)
    internal_size = self._vhdutils._get_internal_vhdx_size_by_file_size(mock.sentinel.vhd_path, new_vhd_sz, fake_vhd_info)
    self.assertIn(type(internal_size), six.integer_types)
    self.assertEqual(expected_max_int_sz, internal_size)