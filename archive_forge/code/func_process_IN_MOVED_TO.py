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
def process_IN_MOVED_TO(self, raw_event):
    """
        Map the source path with the destination path (+ date for
        cleaning).
        """
    watch_ = self._watch_manager.get_watch(raw_event.wd)
    path_ = watch_.path
    dst_path = os.path.normpath(os.path.join(path_, raw_event.name))
    mv_ = self._mv_cookie.get(raw_event.cookie)
    to_append = {'cookie': raw_event.cookie}
    if mv_ is not None:
        self._mv[mv_[0]] = (dst_path, datetime.now())
        to_append['src_pathname'] = mv_[0]
    elif raw_event.mask & IN_ISDIR and watch_.auto_add and (not watch_.exclude_filter(dst_path)):
        self._watch_manager.add_watch(dst_path, watch_.mask, proc_fun=watch_.proc_fun, rec=True, auto_add=True, exclude_filter=watch_.exclude_filter)
    return self.process_default(raw_event, to_append)