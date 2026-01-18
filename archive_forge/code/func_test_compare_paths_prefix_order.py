import errno
import os
import select
import socket
import sys
import tempfile
import time
from io import BytesIO
from .. import errors, osutils, tests, trace, win32utils
from . import features, file_utils, test__walkdirs_win32
from .scenarios import load_tests_apply_scenarios
def test_compare_paths_prefix_order(self):
    self.assertPathCompare('/', '/a')
    self.assertPathCompare('/a', '/b')
    self.assertPathCompare('/b', '/z')
    self.assertPathCompare('/z', '/a/a')
    self.assertPathCompare('/a/b/c', '/d/g')
    self.assertPathCompare('/a/z', '/z/z')
    self.assertPathCompare('/a/c/z', '/a/d/e')
    self.assertPathCompare('', 'a')
    self.assertPathCompare('a', 'b')
    self.assertPathCompare('b', 'z')
    self.assertPathCompare('z', 'a/a')
    self.assertPathCompare('a/b/c', 'd/g')
    self.assertPathCompare('a/z', 'z/z')
    self.assertPathCompare('a/c/z', 'a/d/e')