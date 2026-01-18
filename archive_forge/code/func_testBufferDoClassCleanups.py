import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferDoClassCleanups(self):
    with captured_stdout() as stdout:
        result = unittest.TestResult()
    result.buffer = True

    class Foo(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            print('set up class')
            cls.addClassCleanup(bad_cleanup1)
            cls.addClassCleanup(bad_cleanup2)

        @classmethod
        def tearDownClass(cls):
            print('tear down class')

        def test_foo(self):
            pass
    suite = unittest.TestSuite([Foo('test_foo')])
    suite(result)
    expected_out = '\nStdout:\ntear down class\ndo cleanup2\ndo cleanup1\n'
    self.assertEqual(stdout.getvalue(), expected_out)
    self.assertEqual(len(result.errors), 2)
    description = f'tearDownClass ({strclass(Foo)})'
    test_case, formatted_exc = result.errors[0]
    self.assertEqual(test_case.description, description)
    self.assertIn('ValueError: bad cleanup2', formatted_exc)
    self.assertNotIn('TypeError', formatted_exc)
    self.assertIn(expected_out, formatted_exc)
    test_case, formatted_exc = result.errors[1]
    self.assertEqual(test_case.description, description)
    self.assertIn('TypeError: bad cleanup1', formatted_exc)
    self.assertNotIn('ValueError', formatted_exc)
    self.assertIn(expected_out, formatted_exc)