import collections
import contextlib
import errno
import functools
import os
import sys
import types
@contextlib.contextmanager
def redirect_stderr(new_target):
    original = sys.stderr
    try:
        sys.stderr = new_target
        yield new_target
    finally:
        sys.stderr = original