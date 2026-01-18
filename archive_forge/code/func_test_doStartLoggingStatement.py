from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import CancelledError
from twisted.internet.interfaces import (
from twisted.internet.protocol import (
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.logger import LogLevel, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
def test_doStartLoggingStatement(self):
    """
        L{Factory.doStart} logs that it is starting a factory, followed by
        the L{repr} of the L{Factory} instance that is being started.
        """
    events = []
    globalLogPublisher.addObserver(events.append)
    self.addCleanup(lambda: globalLogPublisher.removeObserver(events.append))
    f = Factory()
    f.doStart()
    self.assertIs(events[0]['factory'], f)
    self.assertEqual(events[0]['log_level'], LogLevel.info)
    self.assertEqual(events[0]['log_format'], 'Starting factory {factory!r}')