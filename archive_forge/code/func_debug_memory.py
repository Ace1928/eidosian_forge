from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def debug_memory(message='', short=True):
    """Write out a memory dump."""
    if sys.platform == 'win32':
        from breezy import win32utils
        win32utils.debug_memory_win32api(message=message, short=short)
    else:
        _debug_memory_proc(message=message, short=short)