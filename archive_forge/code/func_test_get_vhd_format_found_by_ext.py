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
def test_get_vhd_format_found_by_ext(self):
    fake_vhd_path = 'C:\\test.vhd'
    ret_val = self._vhdutils.get_vhd_format(fake_vhd_path)
    self.assertEqual(constants.DISK_FORMAT_VHD, ret_val)