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
def test_from_file_eager_loading(self):
    fd, path = tempfile.mkstemp()
    os.write(fd, _b('some data'))
    os.close(fd)
    content = content_from_file(path, UTF8_TEXT, buffer_now=True)
    os.remove(path)
    self.assertThat(''.join(content.iter_text()), Equals('some data'))