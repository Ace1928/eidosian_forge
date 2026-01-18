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
def flatten_vhd(self, vhd_path):
    base_path, ext = os.path.splitext(vhd_path)
    tmp_path = base_path + '.tmp' + ext
    self.convert_vhd(vhd_path, tmp_path)
    os.unlink(vhd_path)
    os.rename(tmp_path, vhd_path)