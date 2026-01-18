import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def test_logged_messages_go_to_observer(self):
    from testtools.twistedsupport._runtest import _TwistedLogObservers
    messages = []

    class SomeTest(TestCase):

        def test_something(self):
            self.useFixture(_TwistedLogObservers([messages.append]))
            log.msg('foo')
    SomeTest('test_something').run()
    log.msg('bar')
    self.assertThat(messages, MatchesListwise([ContainsDict({'message': Equals(('foo',))})]))