from collections.abc import Sequence
import functools
import inspect
import linecache
import pydoc
import sys
import time
import traceback
import types
from types import TracebackType
from typing import Any, List, Optional, Tuple
import stack_data
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.styles import get_style_by_name
import IPython.utils.colorable as colorable
from IPython import get_ipython
from IPython.core import debugger
from IPython.core.display_trap import DisplayTrap
from IPython.core.excolors import exception_colors
from IPython.utils import PyColorize
from IPython.utils import path as util_path
from IPython.utils import py3compat
from IPython.utils.terminal import get_terminal_size
def structured_traceback(self, etype, value, elist, tb_offset=None, context=5):
    if isinstance(value, SyntaxError) and isinstance(value.filename, str) and isinstance(value.lineno, int):
        linecache.checkcache(value.filename)
        newtext = linecache.getline(value.filename, value.lineno)
        if newtext:
            value.text = newtext
    self.last_syntax_error = value
    return super(SyntaxTB, self).structured_traceback(etype, value, elist, tb_offset=tb_offset, context=context)