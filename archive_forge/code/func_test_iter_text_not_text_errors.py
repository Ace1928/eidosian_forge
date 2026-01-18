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
def test_iter_text_not_text_errors(self):
    content_type = ContentType('foo', 'bar')
    content = Content(content_type, lambda: ['bytes'])
    self.assertThat(content.iter_text, raises_value_error)