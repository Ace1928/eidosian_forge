import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_assert_fails_with_success(self):
    marker = object()
    d = assert_fails_with(defer.succeed(marker), RuntimeError)

    def check_result(failure):
        failure.trap(self.failureException)
        self.assertThat(str(failure.value), Equals(f'RuntimeError not raised ({marker!r} returned)'))
    d.addCallbacks(lambda x: self.fail('Should not have succeeded'), check_result)
    return d