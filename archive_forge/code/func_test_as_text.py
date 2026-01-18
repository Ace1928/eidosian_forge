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
def test_as_text(self):
    content_type = ContentType('text', 'strange', {'charset': 'utf8'})
    content = Content(content_type, lambda: ['bytesê'.encode()])
    self.assertEqual('bytesê', content.as_text())