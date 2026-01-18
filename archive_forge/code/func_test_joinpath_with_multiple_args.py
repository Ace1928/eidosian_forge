import textwrap
import unittest
import warnings
import importlib
import contextlib
import importlib_resources as resources
from ..abc import Traversable
from . import data01
from . import util
from . import _path
from .compat.py39 import os_helper
from .compat.py312 import import_helper
def test_joinpath_with_multiple_args(self):
    files = resources.files(self.data)
    binfile = files.joinpath('subdirectory', 'binary.file')
    self.assertTrue(binfile.is_file())