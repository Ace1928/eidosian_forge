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
def test_eager_read_by_default(self):

    class SomeTest(TestCase):

        def test_foo(self):
            pass
    test = SomeTest('test_foo')
    path = self.make_file('some data')
    attach_file(test, path, name='foo')
    content = test.getDetails()['foo']
    content_file = open(path, 'w')
    content_file.write('new data')
    content_file.close()
    self.assertEqual(''.join(content.iter_text()), 'some data')