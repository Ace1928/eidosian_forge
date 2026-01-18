import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
def test_from_file_with_simple_seek(self):
    f = tempfile.NamedTemporaryFile()
    f.write(_b('some data'))
    f.flush()
    self.addCleanup(f.close)
    content = content_from_file(f.name, UTF8_TEXT, chunk_size=50, seek_offset=5)
    self.assertThat(list(content.iter_bytes()), Equals([_b('data')]))