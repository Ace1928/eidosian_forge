import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def consume_many(self, count):
    self._fill_buffer(count)
    buffer = self._buffer
    if len(buffer) == count:
        ret = list(buffer)
        buffer.clear()
    else:
        ret = []
        while buffer and count:
            ret.append(buffer.popleft())
            count -= 1
    return ret