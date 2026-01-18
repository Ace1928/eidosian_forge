import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def swallow_exception(format_string='', *args, **kwargs):
    """Logs an exception with full traceback.

    If format_string is specified, it is formatted with format(*args, **kwargs), and
    prepended to the exception traceback on a separate line.

    If exc_info is specified, the exception it describes will be logged. Otherwise,
    sys.exc_info() - i.e. the exception being handled currently - will be logged.

    If level is specified, the exception will be logged as a message of that level.
    The default is "error".
    """
    _exception(format_string, *args, **kwargs)