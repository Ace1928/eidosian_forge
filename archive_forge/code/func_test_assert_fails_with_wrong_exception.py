import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_assert_fails_with_wrong_exception(self):
    d = assert_fails_with(defer.maybeDeferred(lambda: 1 / 0), RuntimeError, KeyboardInterrupt)

    def check_result(failure):
        failure.trap(self.failureException)
        lines = str(failure.value).splitlines()
        self.assertThat(lines[:2], Equals(['ZeroDivisionError raised instead of RuntimeError, KeyboardInterrupt:', ' Traceback (most recent call last):']))
    d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
    return d