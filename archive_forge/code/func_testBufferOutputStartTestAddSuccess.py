import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def testBufferOutputStartTestAddSuccess(self):
    real_out = self._real_out
    real_err = self._real_err
    result = unittest.TestResult()
    self.assertFalse(result.buffer)
    result.buffer = True
    self.assertIs(real_out, sys.stdout)
    self.assertIs(real_err, sys.stderr)
    result.startTest(self)
    self.assertIsNot(real_out, sys.stdout)
    self.assertIsNot(real_err, sys.stderr)
    self.assertIsInstance(sys.stdout, io.StringIO)
    self.assertIsInstance(sys.stderr, io.StringIO)
    self.assertIsNot(sys.stdout, sys.stderr)
    out_stream = sys.stdout
    err_stream = sys.stderr
    result._original_stdout = io.StringIO()
    result._original_stderr = io.StringIO()
    print('foo')
    print('bar', file=sys.stderr)
    self.assertEqual(out_stream.getvalue(), 'foo\n')
    self.assertEqual(err_stream.getvalue(), 'bar\n')
    self.assertEqual(result._original_stdout.getvalue(), '')
    self.assertEqual(result._original_stderr.getvalue(), '')
    result.addSuccess(self)
    result.stopTest(self)
    self.assertIs(sys.stdout, result._original_stdout)
    self.assertIs(sys.stderr, result._original_stderr)
    self.assertEqual(result._original_stdout.getvalue(), '')
    self.assertEqual(result._original_stderr.getvalue(), '')
    self.assertEqual(out_stream.getvalue(), '')
    self.assertEqual(err_stream.getvalue(), '')