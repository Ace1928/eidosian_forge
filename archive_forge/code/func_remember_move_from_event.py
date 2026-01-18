from __future__ import with_statement
import os
import errno
import struct
import threading
import ctypes
import ctypes.util
from functools import reduce
from ctypes import c_int, c_char_p, c_uint32
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import UnsupportedLibc
def remember_move_from_event(self, event):
    """
        Save this event as the source event for future MOVED_TO events to
        reference.
        """
    self._moved_from_events[event.cookie] = event