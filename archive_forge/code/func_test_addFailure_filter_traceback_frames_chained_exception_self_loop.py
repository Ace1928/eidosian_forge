import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def test_addFailure_filter_traceback_frames_chained_exception_self_loop(self):

    class Foo(unittest.TestCase):

        def test_1(self):
            pass

    def get_exc_info():
        try:
            loop = Exception('Loop')
            loop.__cause__ = loop
            loop.__context__ = loop
            raise loop
        except:
            return sys.exc_info()
    exc_info_tuple = get_exc_info()
    test = Foo('test_1')
    result = unittest.TestResult()
    result.startTest(test)
    result.addFailure(test, exc_info_tuple)
    result.stopTest(test)
    formatted_exc = result.failures[0][1]
    self.assertEqual(formatted_exc.count('Exception: Loop\n'), 1)