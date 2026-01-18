from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testUnexpectedErrorActuallyLater(self):

    def myiter():
        D = defer.Deferred()
        reactor.callLater(0, D.errback, RuntimeError())
        yield D
    c = task.Cooperator()
    d = c.coiterate(myiter())
    return self.assertFailure(d, RuntimeError)