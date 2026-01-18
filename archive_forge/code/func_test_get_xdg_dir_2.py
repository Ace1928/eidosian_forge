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
@with_environment
def test_get_xdg_dir_2():
    """test_get_xdg_dir_2, check xdg_dir default to ~/.config"""
    reload(path)
    path.get_home_dir = lambda: HOME_TEST_DIR
    os.name = 'posix'
    sys.platform = 'linux2'
    env.pop('IPYTHON_DIR', None)
    env.pop('IPYTHONDIR', None)
    env.pop('XDG_CONFIG_HOME', None)
    cfgdir = os.path.join(path.get_home_dir(), '.config')
    if not os.path.exists(cfgdir):
        os.makedirs(cfgdir)
    assert path.get_xdg_dir() == cfgdir