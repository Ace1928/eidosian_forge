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
def test_from_file_default_type(self):
    content = content_from_file('/nonexistent/path')
    self.assertThat(content.content_type, Equals(UTF8_TEXT))