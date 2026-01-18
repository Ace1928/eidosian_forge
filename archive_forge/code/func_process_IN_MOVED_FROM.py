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
def process_IN_MOVED_FROM(self, raw_event):
    """
        Map the cookie with the source path (+ date for cleaning).
        """
    watch_ = self._watch_manager.get_watch(raw_event.wd)
    path_ = watch_.path
    src_path = os.path.normpath(os.path.join(path_, raw_event.name))
    self._mv_cookie[raw_event.cookie] = (src_path, datetime.now())
    return self.process_default(raw_event, {'cookie': raw_event.cookie})