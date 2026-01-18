import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def test_locals(self):
    program = self.program
    program.testRunner = FakeRunner
    program.parseArgs([None, '--locals'])
    self.assertEqual(True, program.tb_locals)
    program.runTests()
    self.assertEqual(FakeRunner.initArgs, {'buffer': False, 'failfast': False, 'tb_locals': True, 'verbosity': 1, 'warnings': None})