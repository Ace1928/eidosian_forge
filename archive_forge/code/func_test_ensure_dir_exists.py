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
@onlyif_unicode_paths
def test_ensure_dir_exists():
    with TemporaryDirectory() as td:
        d = os.path.join(td, '∂ir')
        path.ensure_dir_exists(d)
        assert os.path.isdir(d)
        path.ensure_dir_exists(d)
        f = os.path.join(td, 'ƒile')
        open(f, 'w', encoding='utf-8').close()
        with pytest.raises(IOError):
            path.ensure_dir_exists(f)