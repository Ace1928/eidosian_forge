import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from contextlib import contextmanager
from io import StringIO
from subprocess import Popen, PIPE
from unittest.mock import patch
from traitlets.config.loader import Config
from IPython.utils.process import get_output_error_code
from IPython.utils.text import list_strings
from IPython.utils.io import temp_pyfile, Tee
from IPython.utils import py3compat
from . import decorators as dec
from . import skipdoctest
def get_ipython_cmd(as_string=False):
    """
    Return appropriate IPython command line name. By default, this will return
    a list that can be used with subprocess.Popen, for example, but passing
    `as_string=True` allows for returning the IPython command as a string.

    Parameters
    ----------
    as_string: bool
        Flag to allow to return the command as a string.
    """
    ipython_cmd = [sys.executable, '-m', 'IPython']
    if as_string:
        ipython_cmd = ' '.join(ipython_cmd)
    return ipython_cmd