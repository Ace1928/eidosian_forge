import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_delayed_call(self):
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    call = reactor.callLater(10, lambda: None)
    results = spinner._clean()
    self.assertThat(results, Equals([call]))
    self.assertThat(call.active(), Equals(False))