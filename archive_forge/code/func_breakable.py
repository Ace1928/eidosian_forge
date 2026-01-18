from contextlib import contextmanager
import datetime
import os
import re
import sys
import types
from collections import deque
from inspect import signature
from io import StringIO
from warnings import warn
from IPython.utils.decorators import undoc
from IPython.utils.py3compat import PYPY
from typing import Dict
def breakable(self, sep=' '):
    """
        Add a breakable separator to the output.  This does not mean that it
        will automatically break here.  If no breaking on this position takes
        place the `sep` is inserted which default to one space.
        """
    width = len(sep)
    group = self.group_stack[-1]
    if group.want_break:
        self.flush()
        self.output.write(self.newline)
        self.output.write(' ' * self.indentation)
        self.output_width = self.indentation
        self.buffer_width = 0
    else:
        self.buffer.append(Breakable(sep, width, self))
        self.buffer_width += width
        self._break_outer_groups()