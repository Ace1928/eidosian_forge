import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
def testSelectedTestNamesFunctionalTest(self):

    def run_unittest(args):
        cmd = [sys.executable, '-E', '-m', 'unittest'] + args
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=os.path.dirname(__file__))
        with p:
            _, stderr = p.communicate()
        return stderr.decode()
    t = '_test_warnings'
    self.assertIn('Ran 7 tests', run_unittest([t]))
    self.assertIn('Ran 7 tests', run_unittest(['-k', 'TestWarnings', t]))
    self.assertIn('Ran 7 tests', run_unittest(['discover', '-p', '*_test*', '-k', 'TestWarnings']))
    self.assertIn('Ran 2 tests', run_unittest(['-k', 'f', t]))
    self.assertIn('Ran 7 tests', run_unittest(['-k', 't', t]))
    self.assertIn('Ran 3 tests', run_unittest(['-k', '*t', t]))
    self.assertIn('Ran 7 tests', run_unittest(['-k', '*test_warnings.*Warning*', t]))
    self.assertIn('Ran 1 test', run_unittest(['-k', '*test_warnings.*warning*', t]))