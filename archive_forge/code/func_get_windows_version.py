import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_windows_version(self):
    """Returns a string representing the host's kernel version."""
    if not HostUtils._windows_version:
        Win32_OperatingSystem = self._conn_cimv2.Win32_OperatingSystem()[0]
        HostUtils._windows_version = Win32_OperatingSystem.Version
    return HostUtils._windows_version