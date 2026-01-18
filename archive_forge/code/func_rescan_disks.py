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
def rescan_disks(self, merge_requests=False):
    """Perform a disk rescan.

        :param merge_requests: If this flag is set and a disk rescan is
                               already pending, we'll just wait for it to
                               finish without issuing a new rescan request.
        """
    if merge_requests:
        rescan_pending = _RESCAN_LOCK.locked()
        if rescan_pending:
            LOG.debug('A disk rescan is already pending. Waiting for it to complete.')
        with _RESCAN_LOCK:
            if not rescan_pending:
                self._rescan_disks()
    else:
        self._rescan_disks()