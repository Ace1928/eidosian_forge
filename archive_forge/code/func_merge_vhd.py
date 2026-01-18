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
def merge_vhd(self, vhd_path, delete_merged_image=True):
    """Merges a VHD/x image into the immediate next parent image."""
    open_params = vdisk_struct.OPEN_VIRTUAL_DISK_PARAMETERS()
    open_params.Version = w_const.OPEN_VIRTUAL_DISK_VERSION_1
    open_params.Version1.RWDepth = 2
    handle = self._open(vhd_path, open_params=ctypes.byref(open_params))
    params = vdisk_struct.MERGE_VIRTUAL_DISK_PARAMETERS()
    params.Version = w_const.MERGE_VIRTUAL_DISK_VERSION_1
    params.Version1.MergeDepth = 1
    self._run_and_check_output(virtdisk.MergeVirtualDisk, handle, 0, ctypes.byref(params), None, cleanup_handle=handle)
    if delete_merged_image:
        os.remove(vhd_path)