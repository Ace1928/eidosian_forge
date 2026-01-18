from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testCallLater(self):
    """
        Test that calls can be scheduled for later with the fake clock and
        hands back an L{IDelayedCall}.
        """
    c = task.Clock()
    call = c.callLater(1, lambda a, b: None, 1, b=2)
    self.assertTrue(interfaces.IDelayedCall.providedBy(call))
    self.assertEqual(call.getTime(), 1)
    self.assertTrue(call.active())