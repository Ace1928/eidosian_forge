from _pydevd_bundle.pydevd_constants import ForkSafeLock, get_global_debugger
import os
import sys
from contextlib import contextmanager
def get_pydb(self):
    return get_global_debugger()