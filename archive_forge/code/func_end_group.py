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
def end_group(self, dedent=0, close=''):
    """End a group. See `begin_group` for more details."""
    self.indentation -= dedent
    group = self.group_stack.pop()
    if not group.breakables:
        self.group_queue.remove(group)
    if close:
        self.text(close)