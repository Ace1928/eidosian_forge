import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferSetUp(self):
    with captured_stdout() as stdout:
        result = unittest.TestResult()
    result.buffer = True

    class Foo(unittest.TestCase):

        def setUp(self):
            print('set up')
            1 / 0

        def test_foo(self):
            pass
    suite = unittest.TestSuite([Foo('test_foo')])
    suite(result)
    expected_out = '\nStdout:\nset up\n'
    self.assertEqual(stdout.getvalue(), expected_out)
    self.assertEqual(len(result.errors), 1)
    description = f'test_foo ({strclass(Foo)}.test_foo)'
    test_case, formatted_exc = result.errors[0]
    self.assertEqual(str(test_case), description)
    self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
    self.assertIn(expected_out, formatted_exc)