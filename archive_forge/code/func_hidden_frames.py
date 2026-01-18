import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
def hidden_frames(self, stack):
    """
        Given an index in the stack return whether it should be skipped.

        This is used in up/down and where to skip frames.
        """
    ip_hide = [self._hidden_predicate(s[0]) for s in stack]
    ip_start = [i for i, s in enumerate(ip_hide) if s == '__ipython_bottom__']
    if ip_start and self._predicates['ipython_internal']:
        ip_hide = [h if i > ip_start[0] else True for i, h in enumerate(ip_hide)]
    return ip_hide