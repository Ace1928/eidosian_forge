from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def gotRootObj(obj):
    failureDeferred = self._addFailingCallbacks(obj.callRemote(method), expected, eb)
    if exc is not None:

        def gotFailure(err):
            self.assertEqual(len(self.flushLoggedErrors(exc)), 1)
            return err
        failureDeferred.addBoth(gotFailure)
    return failureDeferred