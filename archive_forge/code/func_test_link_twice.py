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
def test_link_twice(self):
    dst = self.dst('target')
    path.link_or_copy(self.src, dst)
    path.link_or_copy(self.src, dst)
    self.assert_inode_equal(self.src, dst)
    assert sorted(os.listdir(self.tempdir.name)) == ['src', 'target']