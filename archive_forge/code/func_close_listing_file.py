from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def close_listing_file():
    if threadlocal.cython_errors_listing_file:
        threadlocal.cython_errors_listing_file.close()
        threadlocal.cython_errors_listing_file = None