import os.path
from os.path import abspath
import re
import sys
import types
import pickle
from test import support
from test.support import import_helper
import unittest
import unittest.mock
import unittest.test
def test_module_symlink_ok(self):
    full_path = self.setup_module_clash()
    original_realpath = os.path.realpath
    mod_dir = os.path.abspath('bar')
    expected_dir = os.path.abspath('foo')

    def cleanup():
        os.path.realpath = original_realpath
    self.addCleanup(cleanup)

    def realpath(path):
        if path == os.path.join(mod_dir, 'foo.py'):
            return os.path.join(expected_dir, 'foo.py')
        return path
    os.path.realpath = realpath
    loader = unittest.TestLoader()
    loader.discover(start_dir='foo', pattern='foo.py')