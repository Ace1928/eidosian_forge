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
def attach_virtual_disk(self, vhd_path, read_only=True, detach_on_handle_close=False):
    """Attach a virtual disk image.

        :param vhd_path: the path of the image to attach
        :param read_only: (bool) attach the image in read only mode
        :parma detach_on_handle_close: if set, the image will automatically be
                                       detached when the last image handle is
                                       closed.
        :returns: if 'detach_on_handle_close' is set, it returns a virtual
                  disk image handle that may be closed using the
                  'close' method of this class.
        """
    open_access_mask = w_const.VIRTUAL_DISK_ACCESS_ATTACH_RO if read_only else w_const.VIRTUAL_DISK_ACCESS_ATTACH_RW
    attach_virtual_disk_flag = 0
    if not detach_on_handle_close:
        attach_virtual_disk_flag |= w_const.ATTACH_VIRTUAL_DISK_FLAG_PERMANENT_LIFETIME
    if read_only:
        attach_virtual_disk_flag |= w_const.ATTACH_VIRTUAL_DISK_FLAG_READ_ONLY
    handle = self._open(vhd_path, open_access_mask=open_access_mask)
    self._run_and_check_output(virtdisk.AttachVirtualDisk, handle, None, attach_virtual_disk_flag, 0, None, None, cleanup_handle=handle if not detach_on_handle_close else None)
    if detach_on_handle_close:
        return handle