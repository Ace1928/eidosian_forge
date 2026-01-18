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
def test_from_nonexistent_file(self):
    directory = tempfile.mkdtemp()
    nonexistent = os.path.join(directory, 'nonexistent-file')
    content = content_from_file(nonexistent)
    self.assertThat(content.iter_bytes, raises(IOError))