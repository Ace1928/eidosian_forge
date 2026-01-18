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
def get_attached_virtual_disk_files(self):
    """Retrieve a list of virtual disks attached to the host.

        This doesn't include disks attached to Hyper-V VMs directly.
        """
    disks = self._conn_storage.Msft_Disk(BusType=BUS_FILE_BACKED_VIRTUAL)
    return [dict(location=disk.Location, number=disk.Number, offline=disk.IsOffline, readonly=disk.IsReadOnly) for disk in disks]