import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def test_defaultTest_with_iterable(self):

    class FakeRunner(object):

        def run(self, test):
            self.test = test
            return True
    old_argv = sys.argv
    sys.argv = ['faketest']
    runner = FakeRunner()
    program = unittest.TestProgram(testRunner=runner, exit=False, defaultTest=['unittest.test', 'unittest.test2'], testLoader=self.FooBarLoader())
    sys.argv = old_argv
    self.assertEqual(['unittest.test', 'unittest.test2'], program.testNames)