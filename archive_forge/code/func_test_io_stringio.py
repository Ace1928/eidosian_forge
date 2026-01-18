import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_io_stringio(self):
    s = io.StringIO()
    self.assertEqual(s, unicode_output_stream(s))