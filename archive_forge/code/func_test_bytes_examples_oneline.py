import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_bytes_examples_oneline(self):
    for b, expected, _ in self.bytes_examples:
        actual = text_repr(b, multiline=False)
        self.assertEqual(actual, self.b_prefix + expected)
        self.assertEqual(ast.literal_eval(actual), b)