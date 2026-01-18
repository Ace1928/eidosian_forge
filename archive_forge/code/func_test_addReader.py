from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_addReader(self):
    """
        Adding a reader when there was previously no reader starts up a
        C{LoopingCall}.
        """
    poller = _ContinuousPolling(Clock())
    self.assertIsNone(poller._loop)
    reader = object()
    self.assertFalse(poller.isReading(reader))
    poller.addReader(reader)
    self.assertIsNotNone(poller._loop)
    self.assertTrue(poller._loop.running)
    self.assertIs(poller._loop.clock, poller._reactor)
    self.assertTrue(poller.isReading(reader))