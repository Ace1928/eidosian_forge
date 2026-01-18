from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_addDetails_with_same_name_as_key_from_get_details(self):
    content = self.get_content()

    class Mismatch:

        def describe(self):
            return 'Mismatch'

        def get_details(self):
            return {'foo': content}

    class Matcher:

        def match(self, thing):
            return Mismatch()

        def __str__(self):
            return 'a description'

    class Case(TestCase):

        def test(self):
            self.addDetail('foo', content)
            self.assertThat('foo', Matcher())
    self.assertDetailsProvided(Case('test'), 'addFailure', ['foo', 'foo-1', 'traceback'])