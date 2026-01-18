from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
def test_getDelayedCalls(self):
    """
        Test that we can get a list of all delayed calls
        """
    c = task.Clock()
    call = c.callLater(1, lambda x: None)
    call2 = c.callLater(2, lambda x: None)
    calls = c.getDelayedCalls()
    self.assertEqual({call, call2}, set(calls))