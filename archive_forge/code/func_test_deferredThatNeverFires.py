from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
def test_deferredThatNeverFires(self):
    self.methodCalled = True
    d = defer.Deferred()
    return d