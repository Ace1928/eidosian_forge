from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def testStopOutstanding(self):
    """
        An iterator run with L{Cooperator.coiterate} paused on a L{Deferred}
        yielded by that iterator will fire its own L{Deferred} (the one
        returned by C{coiterate}) when L{Cooperator.stop} is called.
        """
    testControlD = defer.Deferred()
    outstandingD = defer.Deferred()

    def myiter():
        reactor.callLater(0, testControlD.callback, None)
        yield outstandingD
        self.fail()
    c = task.Cooperator()
    d = c.coiterate(myiter())

    def stopAndGo(ign):
        c.stop()
        outstandingD.callback('arglebargle')
    testControlD.addCallback(stopAndGo)
    d.addCallback(self.cbIter)
    d.addErrback(self.ebIter)
    return d.addCallback(lambda result: self.assertEqual(result, self.RESULT))