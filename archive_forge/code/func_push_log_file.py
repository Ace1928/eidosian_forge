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
def push_log_file(to_file, log_format=None, date_format=None):
    """Intercept log and trace messages and send them to a file.

    :param to_file: A file-like object to which messages will be sent.

    :returns: A memento that should be passed to _pop_log_file to restore the
        previously active logging.
    """
    global _trace_file
    new_handler = EncodedStreamHandler(to_file, 'utf-8', level=logging.DEBUG)
    if log_format is None:
        log_format = '%(levelname)8s  %(message)s'
    new_handler.setFormatter(logging.Formatter(log_format, date_format))
    brz_logger = logging.getLogger('brz')
    old_handlers = brz_logger.handlers[:]
    del brz_logger.handlers[:]
    brz_logger.addHandler(new_handler)
    brz_logger.setLevel(logging.DEBUG)
    old_trace_file = _trace_file
    _trace_file = to_file
    return ('log_memento', old_handlers, new_handler, old_trace_file, to_file)