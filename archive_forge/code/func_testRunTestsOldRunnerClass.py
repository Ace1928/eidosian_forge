import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testRunTestsOldRunnerClass(self):
    program = self.program
    FakeRunner.raiseError = 2
    program.testRunner = FakeRunner
    program.verbosity = 'verbosity'
    program.failfast = 'failfast'
    program.buffer = 'buffer'
    program.test = 'test'
    program.runTests()
    self.assertEqual(FakeRunner.initArgs, {})
    self.assertEqual(FakeRunner.test, 'test')
    self.assertIs(program.result, RESULT)