import os
import signal
from testtools.helpers import try_import
from testtools import skipIf
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
def test_leftover_junk_available(self):
    from twisted.internet.protocol import ServerFactory
    reactor = self.make_reactor()
    spinner = self.make_spinner(reactor)
    port = spinner.run(self.make_timeout(), reactor.listenTCP, 0, ServerFactory(), interface='127.0.0.1')
    self.assertThat(spinner.get_junk(), Equals([port]))