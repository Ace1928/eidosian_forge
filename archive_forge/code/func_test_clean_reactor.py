import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_reactor(self):
    reactor = self.make_reactor()
    timeout = self.make_timeout()

    class SomeCase(TestCase):

        def test_cruft(self):
            reactor.callLater(timeout * 10.0, lambda: None)
    test = SomeCase('test_cruft')
    runner = self.make_runner(test, timeout)
    result = self.make_result()
    runner.run(result)
    self.assertThat([event[:2] for event in result._events], Equals([('startTest', test), ('addError', test), ('stopTest', test)]))
    error = result._events[1][2]
    self.assertThat(error, KeysEqual('traceback', 'twisted-log'))