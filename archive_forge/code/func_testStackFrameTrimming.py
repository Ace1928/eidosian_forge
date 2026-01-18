import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testStackFrameTrimming(self):

    class Frame(object):

        class tb_frame(object):
            f_globals = {}
    result = unittest.TestResult()
    self.assertFalse(result._is_relevant_tb_level(Frame))
    Frame.tb_frame.f_globals['__unittest'] = True
    self.assertTrue(result._is_relevant_tb_level(Frame))