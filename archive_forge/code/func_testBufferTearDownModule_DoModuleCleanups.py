import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferTearDownModule_DoModuleCleanups(self):
    with captured_stdout() as stdout:
        result = unittest.TestResult()
    result.buffer = True

    class Foo(unittest.TestCase):

        def test_foo(self):
            pass

    class Module(object):

        @staticmethod
        def setUpModule():
            print('set up module')
            unittest.addModuleCleanup(bad_cleanup1)
            unittest.addModuleCleanup(bad_cleanup2)

        @staticmethod
        def tearDownModule():
            print('tear down module')
            1 / 0
    Foo.__module__ = 'Module'
    sys.modules['Module'] = Module
    self.addCleanup(sys.modules.pop, 'Module')
    suite = unittest.TestSuite([Foo('test_foo')])
    suite(result)
    expected_out = '\nStdout:\ntear down module\ndo cleanup2\ndo cleanup1\n'
    self.assertEqual(stdout.getvalue(), expected_out)
    self.assertEqual(len(result.errors), 2)
    description = 'tearDownModule (Module)'
    test_case, formatted_exc = result.errors[0]
    self.assertEqual(test_case.description, description)
    self.assertIn('ZeroDivisionError: division by zero', formatted_exc)
    self.assertNotIn('ValueError', formatted_exc)
    self.assertNotIn('TypeError', formatted_exc)
    self.assertIn('\nStdout:\ntear down module\n', formatted_exc)
    test_case, formatted_exc = result.errors[1]
    self.assertEqual(test_case.description, description)
    self.assertIn('ValueError: bad cleanup2', formatted_exc)
    self.assertNotIn('ZeroDivisionError', formatted_exc)
    self.assertNotIn('TypeError', formatted_exc)
    self.assertIn(expected_out, formatted_exc)