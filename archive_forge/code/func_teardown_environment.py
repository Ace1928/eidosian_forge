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
def teardown_environment():
    """Restore things that were remembered by the setup_environment function
    """
    oldenv, os.name, sys.platform, path.get_home_dir, IPython.__file__, old_wd = oldstuff
    os.chdir(old_wd)
    reload(path)
    for key in list(env):
        if key not in oldenv:
            del env[key]
    env.update(oldenv)
    if hasattr(sys, 'frozen'):
        del sys.frozen