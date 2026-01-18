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
@dec.skip_if_not_win32
def test_get_long_path_name_win32():
    with TemporaryDirectory() as tmpdir:
        long_path = os.path.join(path.get_long_path_name(tmpdir), 'this is my long path name')
        os.makedirs(long_path)
        short_path = os.path.join(tmpdir, 'THISIS~1')
        evaluated_path = path.get_long_path_name(short_path)
        assert evaluated_path.lower() == long_path.lower()