from __future__ import annotations
import logging # isort:skip
import os
import sys
from os.path import (
from tempfile import NamedTemporaryFile
def default_filename(ext: str) -> str:
    """ Generate a default filename with a given extension, attempting to use
    the filename of the currently running process, if possible.

    If the filename of the current process is not available (or would not be
    writable), then a temporary file with the given extension is returned.

    Args:
        ext (str) : the desired extension for the filename

    Returns:
        str

    Raises:
        RuntimeError
            If the extensions requested is ".py"

    """
    if ext == 'py':
        raise RuntimeError("asked for a default filename with 'py' extension")
    filename = detect_current_filename()
    if filename is None:
        return temp_filename(ext)
    basedir = dirname(filename) or os.getcwd()
    if _no_access(basedir) or _shares_exec_prefix(basedir):
        return temp_filename(ext)
    name, _ = splitext(basename(filename))
    return join(basedir, name + '.' + ext)