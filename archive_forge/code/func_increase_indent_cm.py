import os
import time
from contextlib import contextmanager
from typing import Callable, Optional
@contextmanager
def increase_indent_cm(title=None, color='MAGENTA'):
    global _debug_indent
    if title:
        dbg('Start: ' + title, color=color)
    _debug_indent += 1
    try:
        yield
    finally:
        _debug_indent -= 1
        if title:
            dbg('End: ' + title, color=color)