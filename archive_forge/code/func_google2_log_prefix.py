import logging as _logging
import os as _os
import sys as _sys
import _thread
import time as _time
import traceback as _traceback
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading
from tensorflow.python.util.tf_export import tf_export
def google2_log_prefix(level, timestamp=None, file_and_line=None):
    """Assemble a logline prefix using the google2 format."""
    global _level_names
    now = timestamp or _time.time()
    now_tuple = _time.localtime(now)
    now_microsecond = int(1000000.0 * (now % 1.0))
    filename, line = file_and_line or _GetFileAndLine()
    basename = _os.path.basename(filename)
    severity = 'I'
    if level in _level_names:
        severity = _level_names[level][0]
    s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (severity, now_tuple[1], now_tuple[2], now_tuple[3], now_tuple[4], now_tuple[5], now_microsecond, _get_thread_id(), basename, line)
    return s