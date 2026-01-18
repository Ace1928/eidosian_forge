import logging
import logging.handlers
import os
import socket
import sys
from humanfriendly import coerce_boolean
from humanfriendly.compat import on_macos, on_windows
from coloredlogs import (
def find_syslog_address():
    """
    Find the most suitable destination for system log messages.

    :returns: The pathname of a log device (a string) or an address/port tuple as
              supported by :class:`~logging.handlers.SysLogHandler`.

    On Mac OS X this prefers :data:`LOG_DEVICE_MACOSX`, after that :data:`LOG_DEVICE_UNIX`
    is checked for existence. If both of these device files don't exist the default used
    by :class:`~logging.handlers.SysLogHandler` is returned.
    """
    if sys.platform == 'darwin' and os.path.exists(LOG_DEVICE_MACOSX):
        return LOG_DEVICE_MACOSX
    elif os.path.exists(LOG_DEVICE_UNIX):
        return LOG_DEVICE_UNIX
    else:
        return ('localhost', logging.handlers.SYSLOG_UDP_PORT)