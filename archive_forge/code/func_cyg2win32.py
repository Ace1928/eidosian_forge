import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def cyg2win32(path: str) -> str:
    """Convert a path from Cygwin-native to Windows-native.

    Uses the cygpath utility (part of the Base install) to do the
    actual conversion.  Falls back to returning the original path if
    this fails.

    Handles the default ``/cygdrive`` mount prefix as well as the
    ``/proc/cygdrive`` portable prefix, custom cygdrive prefixes such
    as ``/`` or ``/mnt``, and absolute paths such as ``/usr/src/`` or
    ``/home/username``

    Parameters
    ----------
    path : str
       The path to convert

    Returns
    -------
    converted_path : str
        The converted path

    Notes
    -----
    Documentation for cygpath utility:
    https://cygwin.com/cygwin-ug-net/cygpath.html
    Documentation for the C function it wraps:
    https://cygwin.com/cygwin-api/func-cygwin-conv-path.html

    """
    if sys.platform != 'cygwin':
        return path
    return subprocess.check_output(['/usr/bin/cygpath', '--windows', path], text=True)