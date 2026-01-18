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
@retry_decorator(error_codes=(w_const.ISDSC_SESSION_BUSY, w_const.ISDSC_DEVICE_BUSY_ON_SESSION))
def logout_storage_target(self, target_iqn):
    LOG.debug('Logging out iSCSI target %(target_iqn)s', dict(target_iqn=target_iqn))
    sessions = self._get_iscsi_target_sessions(target_iqn, connected_only=False)
    for session in sessions:
        self._logout_iscsi_target(session.SessionId)
    self._remove_target_persistent_logins(target_iqn)
    self._remove_static_target(target_iqn)