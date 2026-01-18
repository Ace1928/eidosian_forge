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
def test_expectThat_does_not_exit_test(self):

    class Test(TestCase):
        marker = False

        def test(self):
            self.expectThat('foo', Equals('bar'))
            Test.marker = True
    result = Test('test').run()
    self.assertFalse(result.wasSuccessful())
    self.assertTrue(Test.marker)