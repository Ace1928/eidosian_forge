import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
@contextmanager
def delimit(self, start, end):
    """A context manager for preparing the source for expressions. It adds
        *start* to the buffer and enters, after exit it adds *end*."""
    self.write(start)
    yield
    self.write(end)