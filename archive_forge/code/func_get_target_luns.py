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
def get_target_luns(self, target_name):
    sessions = self._get_iscsi_target_sessions(target_name)
    if sessions:
        luns = self._get_iscsi_session_disk_luns(sessions[0].SessionId)
        return luns
    return []