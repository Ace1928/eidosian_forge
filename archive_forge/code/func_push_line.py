from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def push_line(self, line):
    """Push line back onto the line buffer.

        :param line: the line with no trailing newline
        """
    self.lineno -= 1
    self._buffer.append(line + b'\n')