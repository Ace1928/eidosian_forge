import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
def test_assertFailure_noException(self):
    d = defer.succeed(None)
    self.assertFailure(d, ZeroDivisionError)
    d.addCallbacks(lambda x: self.fail('Should have failed'), lambda x: x.trap(self.failureException))
    return d