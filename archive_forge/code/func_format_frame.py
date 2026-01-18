from types import TracebackType
from typing import List, Optional
import tempfile
import traceback
import contextlib
import inspect
import os.path
def format_frame(frame, *, base=None, line=False):
    """
    Format a FrameSummary in a short way, without printing full absolute path or code.

    The idea is the result fits on a single line.
    """
    extra_line = ''
    if line:
        extra_line = f'{frame.line}  # '
    return f'{extra_line}{shorten_filename(frame.filename, base=base)}:{frame.lineno} in {frame.name}'