from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def stopAndGo(ign):
    c.stop()
    outstandingD.callback('arglebargle')