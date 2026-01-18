import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferSetupClass(self):
    with captured_stdout() as stdout:
        result = unittest.TestResult()
    result.buffer = True

    class Foo(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            print('set up class')
            1 / 0

        def test_foo(self):
            pass
    suite = unittest.TestSuite([Foo('test_foo')])
    suite(result)
    expected_out = '\nStdout:\nset up class\n'
    self.assertEqual(stdout.getvalue(), expected_out)
    self.assertEqual(len(result.errors), 1)
    description = f'setUpClass ({strclass(Foo)})'
    test_case, formatted_exc = result.errors[0]
    self.assertEqual(test_case.description, description)
    self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
    self.assertIn(expected_out, formatted_exc)