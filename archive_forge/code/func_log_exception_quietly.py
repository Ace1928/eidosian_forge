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
def log_exception_quietly():
    """Log the last exception to the trace file only.

    Used for exceptions that occur internally and that may be
    interesting to developers but not to users.  For example,
    errors loading plugins.
    """
    import traceback
    mutter(traceback.format_exc())