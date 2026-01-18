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
def test_find_tests_customize_via_package_pattern(self):
    original_listdir = os.listdir

    def restore_listdir():
        os.listdir = original_listdir
    self.addCleanup(restore_listdir)
    original_isfile = os.path.isfile

    def restore_isfile():
        os.path.isfile = original_isfile
    self.addCleanup(restore_isfile)
    original_isdir = os.path.isdir

    def restore_isdir():
        os.path.isdir = original_isdir
    self.addCleanup(restore_isdir)
    self.addCleanup(sys.path.remove, abspath('/foo'))
    vfs = {abspath('/foo'): ['my_package'], abspath('/foo/my_package'): ['__init__.py', 'test_module.py']}

    def list_dir(path):
        return list(vfs[path])
    os.listdir = list_dir
    os.path.isdir = lambda path: not path.endswith('.py')
    os.path.isfile = lambda path: path.endswith('.py')

    class Module(object):
        paths = []
        load_tests_args = []

        def __init__(self, path):
            self.path = path
            self.paths.append(path)
            if path.endswith('test_module'):

                def load_tests(loader, tests, pattern):
                    self.load_tests_args.append((loader, tests, pattern))
                    return [self.path + ' load_tests']
            else:

                def load_tests(loader, tests, pattern):
                    self.load_tests_args.append((loader, tests, pattern))
                    __file__ = '/foo/my_package/__init__.py'
                    this_dir = os.path.dirname(__file__)
                    pkg_tests = loader.discover(start_dir=this_dir, pattern=pattern)
                    return [self.path + ' load_tests', tests] + pkg_tests
            self.load_tests = load_tests

        def __eq__(self, other):
            return self.path == other.path
    loader = unittest.TestLoader()
    loader._get_module_from_name = lambda name: Module(name)
    loader.suiteClass = lambda thing: thing
    loader._top_level_dir = abspath('/foo')
    suite = list(loader._find_tests(abspath('/foo'), 'test*.py'))
    self.assertEqual(suite, [['my_package load_tests', [], ['my_package.test_module load_tests']]])
    self.assertEqual(Module.paths, ['my_package', 'my_package.test_module'])
    self.assertEqual(Module.load_tests_args, [(loader, [], 'test*.py'), (loader, [], 'test*.py')])