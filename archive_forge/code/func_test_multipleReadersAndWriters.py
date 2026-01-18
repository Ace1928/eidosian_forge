from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_multipleReadersAndWriters(self):
    """
        Adding multiple readers and writers results in a single
        C{LoopingCall}.
        """
    poller = _ContinuousPolling(Clock())
    writer = object()
    poller.addWriter(writer)
    self.assertIsNotNone(poller._loop)
    poller.addWriter(object())
    self.assertIsNotNone(poller._loop)
    poller.addReader(object())
    self.assertIsNotNone(poller._loop)
    poller.addReader(object())
    poller.removeWriter(writer)
    self.assertIsNotNone(poller._loop)
    self.assertTrue(poller._loop.running)
    self.assertEqual(len(poller._reactor.getDelayedCalls()), 1)