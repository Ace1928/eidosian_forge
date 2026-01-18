import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_delayed_call_cancelled(self):
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    call = reactor.callLater(10, lambda: None)
    call.cancel()
    results = spinner._clean()
    self.assertThat(results, Equals([]))