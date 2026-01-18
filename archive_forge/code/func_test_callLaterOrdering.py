from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_callLaterOrdering(self):
    """
        Test that the DelayedCall returned is not one previously
        created.
        """
    c = task.Clock()
    call1 = c.callLater(10, lambda a, b: None, 1, b=2)
    call2 = c.callLater(1, lambda a, b: None, 3, b=4)
    self.assertFalse(call1 is call2)