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
def ensure_lun_available(self, target_iqn, target_lun, rescan_attempts=_DEFAULT_RESCAN_ATTEMPTS, retry_interval=0, rescan_disks=True, ensure_mpio_claimed=False):
    for attempt in range(rescan_attempts + 1):
        sessions = self._get_iscsi_target_sessions(target_iqn)
        for session in sessions:
            try:
                sid = session.SessionId
                device = self._get_iscsi_device_from_session(sid, target_lun)
                if not device:
                    continue
                device_number = device.StorageDeviceNumber.DeviceNumber
                device_path = device.LegacyName
                if not device_path or device_number in (None, -1):
                    continue
                if ensure_mpio_claimed and (not self._diskutils.is_mpio_disk(device_number)):
                    LOG.debug('Disk %s was not claimed yet by the MPIO service.', device_path)
                    continue
                return (device_number, device_path)
            except exceptions.ISCSIInitiatorAPIException:
                err_msg = 'Could not find lun %(target_lun)s  for iSCSI target %(target_iqn)s.'
                LOG.exception(err_msg, dict(target_lun=target_lun, target_iqn=target_iqn))
                continue
        if attempt <= rescan_attempts:
            if retry_interval:
                time.sleep(retry_interval)
            if rescan_disks:
                self._diskutils.rescan_disks()
    raise exceptions.ISCSILunNotAvailable(target_lun=target_lun, target_iqn=target_iqn)