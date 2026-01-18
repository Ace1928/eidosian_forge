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
def test_from_stream(self):
    data = io.StringIO('some data')
    content = content_from_stream(data, UTF8_TEXT, chunk_size=2)
    self.assertThat(list(content.iter_bytes()), Equals(['so', 'me', ' d', 'at', 'a']))