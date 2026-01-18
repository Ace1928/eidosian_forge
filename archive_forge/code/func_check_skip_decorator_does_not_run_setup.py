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
def check_skip_decorator_does_not_run_setup(self, decorator, reason):

    class SkippingTest(TestCase):
        setup_ran = False

        def setUp(self):
            super().setUp()
            self.setup_ran = True

        @decorator
        def test_skipped(self):
            self.fail()
    test = SkippingTest('test_skipped')
    self.check_test_does_not_run_setup(test, reason)

    @decorator
    class SkippingTestCase(TestCase):
        setup_ran = False

        def setUp(self):
            super().setUp()
            self.setup_ran = True

        def test_skipped(self):
            self.fail()
    try:
        test = SkippingTestCase('test_skipped')
    except TestSkipped:
        self.fail('TestSkipped raised')
    self.check_test_does_not_run_setup(test, reason)