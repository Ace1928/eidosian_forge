import ctypes
import functools
import inspect
import socket
import time
from oslo_log import log as logging
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.errmsg import iscsierr
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import iscsidsc as iscsi_struct
def get_device_number_and_path(self, target_name, target_lun, fail_if_not_found=False, retry_attempts=10, retry_interval=0.1, rescan_disks=False, ensure_mpio_claimed=False):
    device_number, device_path = (None, None)
    try:
        device_number, device_path = self.ensure_lun_available(target_name, target_lun, rescan_attempts=retry_attempts, retry_interval=retry_interval, rescan_disks=rescan_disks, ensure_mpio_claimed=ensure_mpio_claimed)
    except exceptions.ISCSILunNotAvailable:
        if fail_if_not_found:
            raise
    return (device_number, device_path)