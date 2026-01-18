import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_bogus_encoding_becomes_ascii(self):
    """A stream with a bogus encoding gets ascii/replace strings"""
    sout = _FakeOutputStream()
    sout.encoding = 'bogus'
    unicode_output_stream(sout).write(self.uni)
    self.assertEqual([_b('pa???n')], sout.writelog)