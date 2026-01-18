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
def test_discover_start_dir_is_package_calls_package_load_tests(self):
    vfs = {abspath('/toplevel'): ['startdir'], abspath('/toplevel/startdir'): ['__init__.py']}

    def list_dir(path):
        return list(vfs[path])
    self.addCleanup(setattr, os, 'listdir', os.listdir)
    os.listdir = list_dir
    self.addCleanup(setattr, os.path, 'isfile', os.path.isfile)
    os.path.isfile = lambda path: path.endswith('.py')
    self.addCleanup(setattr, os.path, 'isdir', os.path.isdir)
    os.path.isdir = lambda path: not path.endswith('.py')
    self.addCleanup(sys.path.remove, abspath('/toplevel'))

    class Module(object):
        paths = []
        load_tests_args = []

        def __init__(self, path):
            self.path = path

        def load_tests(self, loader, tests, pattern):
            return ['load_tests called ' + self.path]

        def __eq__(self, other):
            return self.path == other.path
    loader = unittest.TestLoader()
    loader._get_module_from_name = lambda name: Module(name)
    loader.suiteClass = lambda thing: thing
    suite = loader.discover('/toplevel/startdir', top_level_dir='/toplevel')
    self.assertEqual(suite, [['load_tests called startdir']])