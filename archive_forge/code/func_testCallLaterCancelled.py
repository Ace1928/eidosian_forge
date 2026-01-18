from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def testCallLaterCancelled(self):
    """
        Test that calls can be cancelled.
        """
    c = task.Clock()
    call = c.callLater(1, lambda a, b: None, 1, b=2)
    call.cancel()
    self.assertFalse(call.active())