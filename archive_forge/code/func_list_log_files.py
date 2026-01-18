from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def list_log_files(pydevd_debug_file):
    log_files = []
    dirname = os.path.dirname(pydevd_debug_file)
    basename = os.path.basename(pydevd_debug_file)
    if os.path.isdir(dirname):
        name, ext = os.path.splitext(basename)
        for f in os.listdir(dirname):
            if f.startswith(name) and f.endswith(ext):
                log_files.append(os.path.join(dirname, f))
    return log_files