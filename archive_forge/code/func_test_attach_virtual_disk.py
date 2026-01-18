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
@ddt.data({}, {'read_only': False, 'detach_on_handle_close': True})
@ddt.unpack
@mock.patch.object(vhdutils.VHDUtils, '_open')
def test_attach_virtual_disk(self, mock_open, read_only=True, detach_on_handle_close=False):
    ret_val = self._vhdutils.attach_virtual_disk(mock.sentinel.vhd_path, read_only, detach_on_handle_close)
    handle = mock_open.return_value
    self.assertEqual(handle if detach_on_handle_close else None, ret_val)
    exp_access_mask = w_const.VIRTUAL_DISK_ACCESS_ATTACH_RO if read_only else w_const.VIRTUAL_DISK_ACCESS_ATTACH_RW
    mock_open.assert_called_once_with(mock.sentinel.vhd_path, open_access_mask=exp_access_mask)
    self._mock_run.assert_called_once_with(vhdutils.virtdisk.AttachVirtualDisk, handle, None, mock.ANY, 0, None, None, **self._run_args)
    if not detach_on_handle_close:
        self._mock_close.assert_called_once_with(handle)
    else:
        self._mock_close.assert_not_called()
    mock_run_args = self._mock_run.call_args_list[0][0]
    attach_flag = mock_run_args[3]
    self.assertEqual(read_only, bool(attach_flag & w_const.ATTACH_VIRTUAL_DISK_FLAG_READ_ONLY))
    self.assertEqual(not detach_on_handle_close, bool(attach_flag & w_const.ATTACH_VIRTUAL_DISK_FLAG_PERMANENT_LIFETIME))