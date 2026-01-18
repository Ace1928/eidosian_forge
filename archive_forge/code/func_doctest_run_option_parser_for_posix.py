import functools
import os
import platform
import random
import string
import sys
import textwrap
import unittest
from os.path import join as pjoin
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
from IPython.core import debugger
from IPython.testing import decorators as dec
from IPython.testing import tools as tt
from IPython.utils.io import capture_output
import gc
@dec.skip_win32
def doctest_run_option_parser_for_posix():
    """Test option parser in %run (Linux/OSX specific).

    You need double quote to escape glob in POSIX systems:

    In [1]: %run print_argv.py print\\\\*.py
    ['print*.py']

    You can't use quote to escape glob in POSIX systems:

    In [2]: %run print_argv.py 'print*.py'
    ['print_argv.py']

    """