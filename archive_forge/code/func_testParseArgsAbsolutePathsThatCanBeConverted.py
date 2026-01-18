import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testParseArgsAbsolutePathsThatCanBeConverted(self):
    cur_dir = os.getcwd()
    program = self.program

    def _join(name):
        return os.path.join(cur_dir, name)
    argv = ['progname', _join('foo/bar/baz.py'), _join('green\\red.py')]
    self._patch_isfile(argv)
    program.createTests = lambda: None
    program.parseArgs(argv)
    expected = ['foo.bar.baz', 'green.red']
    self.assertEqual(program.testNames, expected)