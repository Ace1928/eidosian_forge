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
def get_vhd_size(self, vhd_path):
    """Return vhd size.

        Returns a dict containing the virtual size, physical size,
        block size and sector size of the vhd.
        """
    size = self.get_vhd_info(vhd_path, [w_const.GET_VIRTUAL_DISK_INFO_SIZE])
    return size