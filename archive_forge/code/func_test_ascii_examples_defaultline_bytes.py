import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_ascii_examples_defaultline_bytes(self):
    for s, one, multi in self.ascii_examples:
        expected = '\n' in s and multi or one
        self.assertEqual(text_repr(_b(s)), self.b_prefix + expected)