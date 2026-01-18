import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from importlib import reload
from os.path import abspath, join
from unittest.mock import patch
import pytest
from tempfile import TemporaryDirectory
import IPython
from IPython import paths
from IPython.testing import decorators as dec
from IPython.testing.decorators import (
from IPython.testing.tools import make_tempfile
from IPython.utils import path
@skip_win32
@with_environment
def test_get_home_dir_3():
    """get_home_dir() uses $HOME if set"""
    env['HOME'] = HOME_TEST_DIR
    home_dir = path.get_home_dir(True)
    assert home_dir == os.path.realpath(env['HOME'])