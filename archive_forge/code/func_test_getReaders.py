from unittest import skipIf
from twisted.internet.error import ConnectionDone
from twisted.internet.posixbase import _ContinuousPolling
from twisted.internet.task import Clock
from twisted.trial.unittest import TestCase
def test_getReaders(self):
    """
        L{_ContinuousPolling.getReaders} returns a list of the read
        descriptors.
        """
    poller = _ContinuousPolling(Clock())
    reader = object()
    poller.addReader(reader)
    self.assertIn(reader, poller.getReaders())