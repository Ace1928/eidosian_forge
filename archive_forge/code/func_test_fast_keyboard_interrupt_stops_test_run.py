import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
@skipIf(os.name != 'posix', 'Sending SIGINT with os.kill is posix only')
def test_fast_keyboard_interrupt_stops_test_run(self):
    SIGINT = getattr(signal, 'SIGINT', None)
    if not SIGINT:
        raise self.skipTest('SIGINT unavailable')

    class SomeCase(TestCase):

        def test_pause(self):
            return defer.Deferred()
    test = SomeCase('test_pause')
    reactor = self.make_reactor()
    timeout = self.make_timeout()
    runner = self.make_runner(test, timeout * 5)
    result = self.make_result()
    reactor.callWhenRunning(os.kill, os.getpid(), SIGINT)
    runner.run(result)
    self.assertThat(result.shouldStop, Equals(True))