import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_log_in_details(self):

    class LogAnError(TestCase):

        def test_something(self):
            log.msg('foo')
            1 / 0
    test = LogAnError('test_something')
    runner = self.make_runner(test, store_twisted_logs=True)
    result = self.make_result()
    runner.run(result)
    self.assertThat(result._events, MatchesEvents(('startTest', test), ('addError', test, {'traceback': Not(Is(None)), 'twisted-log': AsText(EndsWith(' foo\n'))}), ('stopTest', test)))