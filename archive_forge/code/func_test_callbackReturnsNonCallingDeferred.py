from __future__ import annotations
import unittest as pyunit
from twisted.internet import defer
from twisted.python.failure import Failure
from twisted.trial import reporter, unittest, util
from twisted.trial.test import detests
def test_callbackReturnsNonCallingDeferred(self) -> None:
    from twisted.internet import reactor
    call = reactor.callLater(2, reactor.crash)
    result = self.runTest('test_calledButNeverCallback')
    if call.active():
        call.cancel()
    self.assertFalse(result.wasSuccessful())
    assert isinstance(result.errors[0][1], Failure)
    self._wasTimeout(result.errors[0][1])