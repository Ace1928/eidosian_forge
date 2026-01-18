import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def check_min_windows_version(self, major, minor, build=0):
    """Compares the host's kernel version with the given version.

        :returns: True if the host's kernel version is higher or equal to
            the given version.
        """
    version_str = self.get_windows_version()
    return list(map(int, version_str.split('.'))) >= [major, minor, build]