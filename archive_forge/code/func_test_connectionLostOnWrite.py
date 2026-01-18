from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_connectionLostOnWrite(self):
    """
        If a C{doWrite} returns a value indicating disconnection,
        C{connectionLost} is called on it.
        """
    reactor = Clock()
    poller = _ContinuousPolling(reactor)
    desc = Descriptor()
    desc.doWrite = lambda: ConnectionDone()
    poller.addWriter(desc)
    self.assertEqual(desc.events, [])
    reactor.advance(0.001)
    self.assertEqual(desc.events, ['lost'])