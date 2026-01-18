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
def get_vhd_format(self, vhd_path):
    vhd_format = os.path.splitext(vhd_path)[1][1:].upper()
    device_id = DEVICE_ID_MAP.get(vhd_format)
    if not device_id and os.path.exists(vhd_path):
        vhd_format = self._get_vhd_format_by_signature(vhd_path)
    if not vhd_format:
        raise exceptions.VHDException(_('Could not retrieve VHD format: %s') % vhd_path)
    return vhd_format