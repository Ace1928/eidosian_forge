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
def begin_group(self, indent=0, open=''):
    """
        Begin a group.
        The first parameter specifies the indentation for the next line (usually
        the width of the opening text), the second the opening text.  All
        parameters are optional.
        """
    if open:
        self.text(open)
    group = Group(self.group_stack[-1].depth + 1)
    self.group_stack.append(group)
    self.group_queue.enq(group)
    self.indentation += indent