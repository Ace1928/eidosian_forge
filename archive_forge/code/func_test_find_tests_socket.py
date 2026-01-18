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
def test_find_tests_socket(self):
    loader = unittest.TestLoader()
    original_listdir = os.listdir

    def restore_listdir():
        os.listdir = original_listdir
    original_isfile = os.path.isfile

    def restore_isfile():
        os.path.isfile = original_isfile
    original_isdir = os.path.isdir

    def restore_isdir():
        os.path.isdir = original_isdir
    path_lists = [['socket']]
    os.listdir = lambda path: path_lists.pop(0)
    self.addCleanup(restore_listdir)
    os.path.isdir = lambda path: False
    self.addCleanup(restore_isdir)
    os.path.isfile = lambda path: False
    self.addCleanup(restore_isfile)
    loader._get_module_from_name = lambda path: path + ' module'
    orig_load_tests = loader.loadTestsFromModule

    def loadTestsFromModule(module, pattern=None):
        base = orig_load_tests(module, pattern=pattern)
        return base + [module + ' tests']
    loader.loadTestsFromModule = loadTestsFromModule
    loader.suiteClass = lambda thing: thing
    top_level = os.path.abspath('/foo')
    loader._top_level_dir = top_level
    suite = list(loader._find_tests(top_level, 'test*.py'))
    self.assertEqual(suite, [])