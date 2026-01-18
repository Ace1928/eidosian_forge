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
def test_filefind():
    """Various tests for filefind"""
    f = tempfile.NamedTemporaryFile()
    alt_dirs = paths.get_ipython_dir()
    t = path.filefind(f.name, alt_dirs)