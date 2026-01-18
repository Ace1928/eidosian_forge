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
def get_disk_uid_and_uid_type(self, disk_number):
    disk = self._get_disk_by_number(disk_number)
    return (disk.UniqueId, disk.UniqueIdFormat)