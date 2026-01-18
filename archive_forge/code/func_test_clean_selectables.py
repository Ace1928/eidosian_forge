import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_clean_selectables(self):
    from twisted.internet.protocol import ServerFactory
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    port = reactor.listenTCP(0, ServerFactory(), interface='127.0.0.1')
    spinner.run(self.make_timeout(), lambda: None)
    results = spinner.get_junk()
    self.assertThat(results, Equals([port]))