from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_copiedFailureLogging(self):
    """
        Test that a copied failure received from a PB call can be logged
        locally.

        Note: this test needs some serious help: all it really tests is that
        log.err(copiedFailure) doesn't raise an exception.
        """
    d = self.clientFactory.getRootObject()

    def connected(rootObj):
        return rootObj.callRemote('synchronousException')
    d.addCallback(connected)

    def exception(failure):
        log.err(failure)
        errs = self.flushLoggedErrors(SynchronousException)
        self.assertEqual(len(errs), 2)
    d.addErrback(exception)
    self.pump.flush()