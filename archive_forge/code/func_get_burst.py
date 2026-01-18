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
def get_burst(self, timeout=IO_QUEUE_TIMEOUT, burst_timeout=IO_QUEUE_BURST_TIMEOUT, max_size=constants.SERIAL_CONSOLE_BUFFER_SIZE):
    data = self.get(timeout=timeout)
    while data and len(data) <= max_size:
        chunk = self.get(timeout=burst_timeout, continue_on_timeout=False)
        if chunk:
            data += chunk
        else:
            break
    return data