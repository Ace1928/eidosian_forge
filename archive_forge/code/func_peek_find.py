import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def peek_find(self, predicate, limit=None):
    buffer = self._buffer
    i = 0
    while limit is None or i < limit:
        if i >= len(buffer):
            self._fill_buffer(i + 5)
            if i >= len(buffer):
                return None
        v = buffer[i]
        if predicate(v):
            return i + 1
        i += 1
    return None