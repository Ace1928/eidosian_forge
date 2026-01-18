import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testParseArgsFileNames(self):
    program = self.program
    argv = ['progname', 'foo.py', 'bar.Py', 'baz.PY', 'wing.txt']
    self._patch_isfile(argv)
    program.createTests = lambda: None
    program.parseArgs(argv)
    expected = ['foo', 'bar', 'baz', 'wing.txt']
    self.assertEqual(program.testNames, expected)