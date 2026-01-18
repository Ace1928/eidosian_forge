import ast
import io
import sys
import traceback
import testtools
from testtools.compat import (
from testtools.matchers import (
def test_io_bytesio(self):
    bytes_io = io.BytesIO()
    self.assertThat(bytes_io, Not(Is(unicode_output_stream(bytes_io))))
    unicode_output_stream(bytes_io).write('foo')