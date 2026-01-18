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
def test_find_tests_default_calls_package_load_tests(self):
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
    directories = ['a_directory', 'test_directory', 'test_directory2']
    path_lists = [directories, [], [], []]
    os.listdir = lambda path: path_lists.pop(0)
    self.addCleanup(restore_listdir)
    os.path.isdir = lambda path: True
    self.addCleanup(restore_isdir)
    os.path.isfile = lambda path: os.path.basename(path) not in directories
    self.addCleanup(restore_isfile)

    class Module(object):
        paths = []
        load_tests_args = []

        def __init__(self, path):
            self.path = path
            self.paths.append(path)
            if os.path.basename(path) == 'test_directory':

                def load_tests(loader, tests, pattern):
                    self.load_tests_args.append((loader, tests, pattern))
                    return [self.path + ' load_tests']
                self.load_tests = load_tests

        def __eq__(self, other):
            return self.path == other.path
    loader._get_module_from_name = lambda name: Module(name)
    orig_load_tests = loader.loadTestsFromModule

    def loadTestsFromModule(module, pattern=None):
        base = orig_load_tests(module, pattern=pattern)
        return base + [module.path + ' module tests']
    loader.loadTestsFromModule = loadTestsFromModule
    loader.suiteClass = lambda thing: thing
    loader._top_level_dir = '/foo'
    suite = list(loader._find_tests('/foo', 'test*.py'))
    self.assertEqual(suite, [['a_directory module tests'], ['test_directory load_tests', 'test_directory module tests'], ['test_directory2 module tests']])
    self.assertEqual(Module.paths, ['a_directory', 'test_directory', 'test_directory2'])
    self.assertEqual(Module.load_tests_args, [(loader, [], 'test*.py')])