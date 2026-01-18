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
def remove_watch(self, path):
    """
        Removes a watch for the given path.

        :param path:
            Path string for which the watch will be removed.
        """
    with self._lock:
        wd = self._wd_for_path.pop(path)
        del self._path_for_wd[wd]
        if inotify_rm_watch(self._inotify_fd, wd) == -1:
            Inotify._raise_error()