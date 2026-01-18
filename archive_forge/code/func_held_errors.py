from __future__ import absolute_import
import sys
from contextlib import contextmanager
from ..Utils import open_new_file
from . import DebugFlags
from . import Options
def held_errors():
    return threadlocal.cython_errors_stack[-1]