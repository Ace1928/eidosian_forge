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
def test_command_line_handling_parseArgs(self):
    program = TestableTestProgram()
    args = []
    program._do_discovery = args.append
    program.parseArgs(['something', 'discover'])
    self.assertEqual(args, [[]])
    args[:] = []
    program.parseArgs(['something', 'discover', 'foo', 'bar'])
    self.assertEqual(args, [['foo', 'bar']])