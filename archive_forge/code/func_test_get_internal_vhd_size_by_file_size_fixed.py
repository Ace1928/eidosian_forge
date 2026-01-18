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
def test_get_internal_vhd_size_by_file_size_fixed(self):
    root_vhd_size = 1 << 30
    real_size = self._mocked_get_internal_vhd_size(root_vhd_size=root_vhd_size, vhd_type=constants.VHD_TYPE_FIXED)
    expected_vhd_size = root_vhd_size - 512
    self.assertEqual(expected_vhd_size, real_size)