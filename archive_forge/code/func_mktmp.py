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
def mktmp(self, src, ext='.py'):
    """Make a valid python temp file."""
    fname = temp_pyfile(src, ext)
    if not hasattr(self, 'tmps'):
        self.tmps = []
    self.tmps.append(fname)
    self.fname = fname