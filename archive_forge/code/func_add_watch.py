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
def add_watch(self, path):
    """
        Adds a watch for the given path.

        :param path:
            Path to begin monitoring.
        """
    with self._lock:
        self._add_watch(path, self._event_mask)