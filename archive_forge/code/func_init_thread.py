from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def init_thread():
    threadlocal.cython_errors_count = 0
    threadlocal.cython_errors_listing_file = None
    threadlocal.cython_errors_echo_file = None
    threadlocal.cython_errors_warn_once_seen = set()
    threadlocal.cython_errors_stack = []