import ctypes
import os
import re
import threading
from collections.abc import Iterable
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils import pathutils
from os_win.utils import win32utils
from os_win.utils.winapi import libs as w_lib
def set_disk_online(self, disk_number):
    disk = self._get_disk_by_number(disk_number)
    err_code = disk.Online()[1]
    if err_code:
        err_msg = _("Failed to bring disk '%(disk_number)s' online. Error code: %(err_code)s.") % dict(disk_number=disk_number, err_code=err_code)
        raise exceptions.DiskUpdateError(message=err_msg)