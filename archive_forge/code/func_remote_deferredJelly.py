from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def remote_deferredJelly(self):
    d = defer.Deferred()
    d.addCallback(self.raiseJelly)
    d.callback(None)
    return d