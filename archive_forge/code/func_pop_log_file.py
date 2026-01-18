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
def pop_log_file(entry):
    """Undo changes to logging/tracing done by _push_log_file.

    This flushes, but does not close the trace file (so that anything that was
    in it is output.

    Takes the memento returned from _push_log_file."""
    magic, old_handlers, new_handler, old_trace_file, new_trace_file = entry
    global _trace_file
    _trace_file = old_trace_file
    brz_logger = logging.getLogger('brz')
    brz_logger.removeHandler(new_handler)
    new_handler.close()
    brz_logger.handlers = old_handlers
    if new_trace_file is not None:
        new_trace_file.flush()