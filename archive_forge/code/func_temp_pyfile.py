import atexit
import os
import sys
import tempfile
from pathlib import Path
from warnings import warn
from IPython.utils.decorators import undoc
from .capture import CapturedIO, capture_output
def temp_pyfile(src, ext='.py'):
    """Make a temporary python file, return filename and filehandle.

    Parameters
    ----------
    src : string or list of strings (no need for ending newlines if list)
        Source code to be written to the file.
    ext : optional, string
        Extension for the generated file.

    Returns
    -------
    (filename, open filehandle)
        It is the caller's responsibility to close the open file and unlink it.
    """
    fname = tempfile.mkstemp(ext)[1]
    with open(Path(fname), 'w', encoding='utf-8') as f:
        f.write(src)
        f.flush()
    return fname