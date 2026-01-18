from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def release_errors(ignore=False):
    held_errors = threadlocal.cython_errors_stack.pop()
    if not ignore:
        for err in held_errors:
            report_error(err)