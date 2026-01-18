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
def test_detect_module_clash(self):
    full_path = self.setup_module_clash()
    loader = unittest.TestLoader()
    mod_dir = os.path.abspath('bar')
    expected_dir = os.path.abspath('foo')
    msg = re.escape("'foo' module incorrectly imported from %r. Expected %r. Is this module globally installed?" % (mod_dir, expected_dir))
    self.assertRaisesRegex(ImportError, '^%s$' % msg, loader.discover, start_dir='foo', pattern='foo.py')
    self.assertEqual(sys.path[0], full_path)