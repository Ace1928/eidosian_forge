from __future__ import absolute_import, print_function, unicode_literals
from contextlib import contextmanager
import pybtex.io
import six
def set_strict_mode(enable=True):
    global strict
    strict = enable