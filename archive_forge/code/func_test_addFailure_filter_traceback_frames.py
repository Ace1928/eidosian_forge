import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def test_addFailure_filter_traceback_frames(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            pass
    test = Foo('test_1')

    def get_exc_info():
        try:
            test.fail('foo')
        except:
            return sys.exc_info()
    exc_info_tuple = get_exc_info()
    full_exc = traceback.format_exception(*exc_info_tuple)
    result = unittest.TestResult()
    result.startTest(test)
    result.addFailure(test, exc_info_tuple)
    result.stopTest(test)
    formatted_exc = result.failures[0][1]
    dropped = [l for l in full_exc if l not in formatted_exc]
    self.assertEqual(len(dropped), 1)
    self.assertIn('raise self.failureException(msg)', dropped[0])