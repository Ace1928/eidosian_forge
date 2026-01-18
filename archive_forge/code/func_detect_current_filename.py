from __future__ import annotations
import logging # isort:skip
import os
import sys
from os.path import (
from tempfile import NamedTemporaryFile
def detect_current_filename() -> str | None:
    """ Attempt to return the filename of the currently running Python process

    Returns None if the filename cannot be detected.
    """
    import inspect
    filename = None
    frame = inspect.currentframe()
    if frame is not None:
        try:
            while frame.f_back and frame.f_globals.get('name') != '__main__':
                frame = frame.f_back
            filename = frame.f_globals.get('__file__')
        finally:
            del frame
    return filename