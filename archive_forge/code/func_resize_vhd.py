import ctypes
import os
import struct
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import units
from os_win._i18n import _
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import virtdisk as vdisk_struct
from os_win.utils.winapi import wintypes
def resize_vhd(self, vhd_path, new_max_size, is_file_max_size=True, validate_new_size=True):
    if is_file_max_size:
        new_internal_max_size = self.get_internal_vhd_size_by_file_size(vhd_path, new_max_size)
    else:
        new_internal_max_size = new_max_size
    if validate_new_size:
        if not self._check_resize_needed(vhd_path, new_internal_max_size):
            return
    self._resize_vhd(vhd_path, new_internal_max_size)