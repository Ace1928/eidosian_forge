import sys
import threading
import os
import select
import struct
import fcntl
import errno
import termios
import array
import logging
import atexit
from collections import deque
from datetime import datetime, timedelta
import time
import re
import asyncore
import glob
import locale
import subprocess
def process_IN_MOVE_SELF(self, raw_event):
    """
        STATUS: the following bug has been fixed in recent kernels (FIXME:
        which version ?). Now it raises IN_DELETE_SELF instead.

        Old kernels were bugged, this event raised when the watched item
        were moved, so we had to update its path, but under some circumstances
        it was impossible: if its parent directory and its destination
        directory wasn't watched. The kernel (see include/linux/fsnotify.h)
        doesn't bring us enough informations like the destination path of
        moved items.
        """
    watch_ = self._watch_manager.get_watch(raw_event.wd)
    src_path = watch_.path
    mv_ = self._mv.get(src_path)
    if mv_:
        dest_path = mv_[0]
        watch_.path = dest_path
        src_path += os.path.sep
        src_path_len = len(src_path)
        for w in self._watch_manager.watches.values():
            if w.path.startswith(src_path):
                w.path = os.path.join(dest_path, w.path[src_path_len:])
    else:
        log.error("The pathname '%s' of this watch %s has probably changed and couldn't be updated, so it cannot be trusted anymore. To fix this error move directories/files only between watched parents directories, in this case e.g. put a watch on '%s'.", watch_.path, watch_, os.path.normpath(os.path.join(watch_.path, os.path.pardir)))
        if not watch_.path.endswith('-unknown-path'):
            watch_.path += '-unknown-path'
    return self.process_default(raw_event)