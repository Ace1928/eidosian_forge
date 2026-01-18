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
def test_addDetailUniqueName_works(self):
    content = self.get_content()

    class Case(TestCase):

        def test(self):
            self.addDetailUniqueName('foo', content)
            self.addDetailUniqueName('foo', content)
    self.assertDetailsProvided(Case('test'), 'addSuccess', ['foo', 'foo-1'])