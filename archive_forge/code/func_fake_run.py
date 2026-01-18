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
def fake_run(func, handle, disk_path_sz_p, disk_path, **kwargs):
    disk_path_sz = ctypes.cast(disk_path_sz_p, wintypes.PULONG).contents.value
    self.assertEqual(w_const.MAX_PATH, disk_path_sz)
    disk_path.value = fake_drive_path