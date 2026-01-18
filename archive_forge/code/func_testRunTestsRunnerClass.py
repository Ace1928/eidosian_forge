import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testRunTestsRunnerClass(self):
    program = self.program
    program.testRunner = FakeRunner
    program.verbosity = 'verbosity'
    program.failfast = 'failfast'
    program.buffer = 'buffer'
    program.warnings = 'warnings'
    program.runTests()
    self.assertEqual(FakeRunner.initArgs, {'verbosity': 'verbosity', 'failfast': 'failfast', 'buffer': 'buffer', 'tb_locals': False, 'warnings': 'warnings'})
    self.assertEqual(FakeRunner.test, 'test')
    self.assertIs(program.result, RESULT)