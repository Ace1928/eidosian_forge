from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def open_listing_file(path, echo_to_stderr=True):
    if path is not None:
        threadlocal.cython_errors_listing_file = open_new_file(path)
    else:
        threadlocal.cython_errors_listing_file = None
    if echo_to_stderr:
        threadlocal.cython_errors_echo_file = sys.stderr
    else:
        threadlocal.cython_errors_echo_file = None
    threadlocal.cython_errors_count = 0