from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
def shorten_filename(fn, *, base=None):
    """Shorten a source filepath, with the assumption that torch/ subdirectories don't need to be shown to user."""
    if base is None:
        base = os.path.dirname(os.path.dirname(__file__))
    try:
        prefix = os.path.commonpath([fn, base])
    except ValueError:
        return fn
    else:
        return fn[len(prefix) + 1:]