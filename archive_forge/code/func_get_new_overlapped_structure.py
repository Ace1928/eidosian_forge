import ctypes
import struct
from eventlet import patcher
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi import wintypes
def get_new_overlapped_structure(self):
    """Structure used for asynchronous IO operations."""
    hEvent = self._create_event()
    overlapped_structure = wintypes.OVERLAPPED()
    overlapped_structure.hEvent = hEvent
    return overlapped_structure