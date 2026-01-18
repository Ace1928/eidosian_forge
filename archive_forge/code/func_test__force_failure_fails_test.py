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
def test__force_failure_fails_test(self):

    class Test(TestCase):

        def test_foo(self):
            self.force_failure = True
            self.remaining_code_run = True
    test = Test('test_foo')
    result = test.run()
    self.assertFalse(result.wasSuccessful())
    self.assertTrue(test.remaining_code_run)