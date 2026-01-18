from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
@lru_cache
def find_tex_file(filename):
    """
    Find a file in the texmf tree using kpathsea_.

    The kpathsea library, provided by most existing TeX distributions, both
    on Unix-like systems and on Windows (MikTeX), is invoked via a long-lived
    luatex process if luatex is installed, or via kpsewhich otherwise.

    .. _kpathsea: https://www.tug.org/kpathsea/

    Parameters
    ----------
    filename : str or path-like

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if isinstance(filename, bytes):
        filename = filename.decode('utf-8', errors='replace')
    try:
        lk = _LuatexKpsewhich()
    except FileNotFoundError:
        lk = None
    if lk:
        path = lk.search(filename)
    else:
        if sys.platform == 'win32':
            kwargs = {'env': {**os.environ, 'command_line_encoding': 'utf-8'}, 'encoding': 'utf-8'}
        else:
            kwargs = {'encoding': sys.getfilesystemencoding(), 'errors': 'surrogateescape'}
        try:
            path = cbook._check_and_log_subprocess(['kpsewhich', filename], _log, **kwargs).rstrip('\n')
        except (FileNotFoundError, RuntimeError):
            path = None
    if path:
        return path
    else:
        raise FileNotFoundError(f"Matplotlib's TeX implementation searched for a file named {filename!r} in your texmf tree, but could not find it")