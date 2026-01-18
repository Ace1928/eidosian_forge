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
@functools.lru_cache()
def count_lines_in_py_file(filename: str) -> int:
    """
    Given a filename, returns the number of lines in the file
    if it ends with the extension ".py". Otherwise, returns 0.
    """
    if not filename.endswith('.py'):
        return 0
    else:
        try:
            with open(filename, 'r') as file:
                s = sum((1 for line in file))
        except UnicodeError:
            return 0
    return s
    '\n    Given a frame object, returns the total number of lines in the file\n    if the filename ends with the extension ".py". Otherwise, returns 0.\n    '