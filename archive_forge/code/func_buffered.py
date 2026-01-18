import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
@contextmanager
def buffered(self, buffer=None):
    if buffer is None:
        buffer = []
    original_source = self._source
    self._source = buffer
    yield buffer
    self._source = original_source