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
def detach_virtual_disk(self, vhd_path):
    if not os.path.exists(vhd_path):
        LOG.debug('Image %s could not be found. Skipping detach.', vhd_path)
        return
    open_access_mask = w_const.VIRTUAL_DISK_ACCESS_DETACH
    try:
        handle = self._open(vhd_path, open_access_mask=open_access_mask)
    except Exception as exc:
        with excutils.save_and_reraise_exception() as ctxt:
            if not self.is_virtual_disk_file_attached(vhd_path):
                LOG.info("The following image is not currently attached to the local host: '%(vhd_path)s'. Ignoring exception encountered while attempting to open and disconnect it: %(exc)s", dict(vhd_path=vhd_path, exc=exc))
                ctxt.reraise = False
                return
    ret_val = self._run_and_check_output(virtdisk.DetachVirtualDisk, handle, 0, 0, ignored_error_codes=[w_const.ERROR_NOT_READY], cleanup_handle=handle)
    if ret_val == w_const.ERROR_NOT_READY:
        LOG.debug('Image %s was not attached.', vhd_path)