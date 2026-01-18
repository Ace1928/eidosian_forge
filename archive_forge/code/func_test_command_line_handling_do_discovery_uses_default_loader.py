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
def test_command_line_handling_do_discovery_uses_default_loader(self):
    program = object.__new__(unittest.TestProgram)
    program._initArgParsers()

    class Loader(object):
        args = []

        def discover(self, start_dir, pattern, top_level_dir):
            self.args.append((start_dir, pattern, top_level_dir))
            return 'tests'
    program.testLoader = Loader()
    program._do_discovery(['-v'])
    self.assertEqual(Loader.args, [('.', 'test*.py', None)])