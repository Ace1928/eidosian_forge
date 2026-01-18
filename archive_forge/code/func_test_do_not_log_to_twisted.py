import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_do_not_log_to_twisted(self):
    messages = []
    publisher, _ = _get_global_publisher_and_observers()
    publisher.addObserver(messages.append)
    self.addCleanup(publisher.removeObserver, messages.append)

    class LogSomething(TestCase):

        def test_something(self):
            log.msg('foo')
    test = LogSomething('test_something')
    runner = self.make_runner(test, suppress_twisted_logging=True)
    result = self.make_result()
    runner.run(result)
    self.assertThat(messages, Equals([]))