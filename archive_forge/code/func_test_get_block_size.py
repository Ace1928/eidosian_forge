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
@mock.patch.object(vhdutils.VHDUtils, '_get_vhdx_metadata_size_and_offset')
def test_get_block_size(self, mock_get_md_sz_and_offset):
    mock_get_md_sz_and_offset.return_value = (mock.sentinel.md_sz, 1024)
    fake_block_size = bytearray(b'\x01\x00\x00\x00')
    fake_offset = bytearray(b'\x02\x00\x00\x00')
    mock_handle = self._get_mock_file_handle(fake_offset, fake_block_size)
    block_size = self._vhdutils._get_vhdx_block_size(mock_handle)
    self.assertEqual(block_size, 1)