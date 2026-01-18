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
def test_command_line_handling_discover_by_default_with_options(self):
    program = TestableTestProgram()
    args = []
    program._do_discovery = args.append
    program.parseArgs(['something', '-v', '-b', '-v', '-c', '-f'])
    self.assertEqual(args, [[]])
    self.assertEqual(program.verbosity, 2)
    self.assertIs(program.buffer, True)
    self.assertIs(program.catchbreak, True)
    self.assertIs(program.failfast, True)