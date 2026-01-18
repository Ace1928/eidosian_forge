from _pydevd_bundle.pydevd_constants import DebugInfoHolder, SHOW_COMPILE_CYTHON_COMMAND_LINE, NULL, LOG_TIME, \
from contextlib import contextmanager
import traceback
import os
import sys
import time
def exception_once(msg, *args):
    try:
        if args:
            message = msg % args
        else:
            message = str(msg)
    except:
        message = '%s - %s' % (msg, args)
    if message not in _LoggingGlobals._warn_once_map:
        _LoggingGlobals._warn_once_map[message] = True
        exception(message)