import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testFailFastSetByRunner(self):
    stream = BufferedWriter()
    runner = unittest.TextTestRunner(stream=stream, failfast=True)

    def test(result):
        self.assertTrue(result.failfast)
    result = runner.run(test)
    stream.flush()
    self.assertTrue(stream.getvalue().endswith('\n\nOK\n'))