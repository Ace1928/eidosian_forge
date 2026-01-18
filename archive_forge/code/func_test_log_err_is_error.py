import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_log_err_is_error(self):

    class LogAnError(TestCase):

        def test_something(self):
            try:
                1 / 0
            except ZeroDivisionError:
                f = failure.Failure()
            log.err(f)
    test = LogAnError('test_something')
    runner = self.make_runner(test, store_twisted_logs=False)
    result = self.make_result()
    runner.run(result)
    self.assertThat(result._events, MatchesEvents(('startTest', test), ('addError', test, {'logged-error': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))