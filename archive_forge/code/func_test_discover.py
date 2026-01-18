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
def test_discover(self):
    loader = unittest.TestLoader()
    original_isfile = os.path.isfile
    original_isdir = os.path.isdir

    def restore_isfile():
        os.path.isfile = original_isfile
    os.path.isfile = lambda path: False
    self.addCleanup(restore_isfile)
    orig_sys_path = sys.path[:]

    def restore_path():
        sys.path[:] = orig_sys_path
    self.addCleanup(restore_path)
    full_path = os.path.abspath(os.path.normpath('/foo'))
    with self.assertRaises(ImportError):
        loader.discover('/foo/bar', top_level_dir='/foo')
    self.assertEqual(loader._top_level_dir, full_path)
    self.assertIn(full_path, sys.path)
    os.path.isfile = lambda path: True
    os.path.isdir = lambda path: True

    def restore_isdir():
        os.path.isdir = original_isdir
    self.addCleanup(restore_isdir)
    _find_tests_args = []

    def _find_tests(start_dir, pattern):
        _find_tests_args.append((start_dir, pattern))
        return ['tests']
    loader._find_tests = _find_tests
    loader.suiteClass = str
    suite = loader.discover('/foo/bar/baz', 'pattern', '/foo/bar')
    top_level_dir = os.path.abspath('/foo/bar')
    start_dir = os.path.abspath('/foo/bar/baz')
    self.assertEqual(suite, "['tests']")
    self.assertEqual(loader._top_level_dir, top_level_dir)
    self.assertEqual(_find_tests_args, [(start_dir, 'pattern')])
    self.assertIn(top_level_dir, sys.path)