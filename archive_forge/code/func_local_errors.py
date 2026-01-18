from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
@contextmanager
def local_errors(ignore=False):
    errors = hold_errors()
    try:
        yield errors
    finally:
        release_errors(ignore=ignore)