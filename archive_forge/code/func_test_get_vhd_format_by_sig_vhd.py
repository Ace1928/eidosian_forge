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
def test_get_vhd_format_by_sig_vhd(self):
    read_data = ('notthesig', vhdutils.VHD_SIGNATURE)
    mock_open = self._mock_open(read_data=read_data, curr_f_pos=1024)
    fmt = self._vhdutils._get_vhd_format_by_signature(mock.sentinel.vhd_path)
    self.assertEqual(constants.DISK_FORMAT_VHD, fmt)
    mock_open.return_value.seek.assert_has_calls([mock.call(0, 2), mock.call(-512, 2)])