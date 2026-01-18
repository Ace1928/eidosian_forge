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
def test_expectThat_adds_detail(self):

    class Test(TestCase):

        def test(self):
            self.expectThat('foo', Equals('bar'))
    test = Test('test')
    test.run()
    details = test.getDetails()
    self.assertIn('Failed expectation', details)