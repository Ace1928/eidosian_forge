import collections
import collections.abc
import logging
import sys
import textwrap
from abc import ABC
def peek_at(self, tokens_ahead):
    self._fill_buffer(tokens_ahead)
    return self._buffer[tokens_ahead - 1] if len(self._buffer) >= tokens_ahead else None