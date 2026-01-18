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
def test_from_stream_eager_loading(self):
    fd, path = tempfile.mkstemp()
    self.addCleanup(os.remove, path)
    self.addCleanup(os.close, fd)
    os.write(fd, _b('some data'))
    stream = open(path, 'rb')
    self.addCleanup(stream.close)
    content = content_from_stream(stream, UTF8_TEXT, buffer_now=True)
    os.write(fd, _b('more data'))
    self.assertThat(''.join(content.iter_text()), Equals('some data'))