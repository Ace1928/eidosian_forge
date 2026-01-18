import sys
from . import *
from contextlib import contextmanager
from string import printable
def buffer_writer(self, text):
    self._buffer.append(text)