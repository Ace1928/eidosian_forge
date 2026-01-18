import os
import signal
from testtools import (
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.runtest import RunTest
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import (
from ._helpers import NeedsTwistedTestCase
def it_pours(self):
    self.addCleanup(lambda: 3 / 0)
    from twisted.internet.protocol import ServerFactory
    reactor.listenTCP(0, ServerFactory(), interface='127.0.0.1')
    defer.maybeDeferred(lambda: 2 / 0)
    raise RuntimeError('Excess precipitation')