from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_getDelayedCallsEmpty(self):
    """
        Test that we get an empty list from getDelayedCalls on a newly
        constructed Clock.
        """
    c = task.Clock()
    self.assertEqual(c.getDelayedCalls(), [])