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
def get_virtual_disk_physical_path(self, vhd_path):
    """Returns the physical disk path for an attached disk image.

        :param vhd_path: an attached disk image path.
        :returns: the mount path of the specified image, in the form of
                  \\\\.\\PhysicalDriveX.
        """
    open_flag = w_const.OPEN_VIRTUAL_DISK_FLAG_NO_PARENTS
    open_access_mask = w_const.VIRTUAL_DISK_ACCESS_GET_INFO | w_const.VIRTUAL_DISK_ACCESS_DETACH
    handle = self._open(vhd_path, open_flag=open_flag, open_access_mask=open_access_mask)
    disk_path = (ctypes.c_wchar * w_const.MAX_PATH)()
    disk_path_sz = wintypes.ULONG(w_const.MAX_PATH)
    self._run_and_check_output(virtdisk.GetVirtualDiskPhysicalPath, handle, ctypes.byref(disk_path_sz), disk_path, cleanup_handle=handle)
    return disk_path.value