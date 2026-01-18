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
def doctest_run_option_parser():
    """Test option parser in %run.

    In [1]: %run print_argv.py
    []

    In [2]: %run print_argv.py print*.py
    ['print_argv.py']

    In [3]: %run -G print_argv.py print*.py
    ['print*.py']

    """