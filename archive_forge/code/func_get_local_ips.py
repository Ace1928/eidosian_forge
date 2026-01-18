import socket
from oslo_log import log as logging
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.winapi import libs as w_lib
def get_local_ips(self):
    """Returns the list of locally assigned IPs."""
    hostname = socket.gethostname()
    return _utils.get_ips(hostname)