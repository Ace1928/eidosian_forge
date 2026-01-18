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
def get_device_number_from_device_name(self, device_name):
    matches = self._phys_dev_name_regex.findall(device_name)
    if matches:
        return matches[0]
    err_msg = _('Could not find device number for device: %s')
    raise exceptions.DiskNotFound(err_msg % device_name)