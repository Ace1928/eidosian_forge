from __future__ import annotations
from twisted.internet import defer, reactor, threads
from twisted.python.failure import Failure
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
def test_calledButNeverCallback(self):
    d = defer.Deferred()

    def neverFire(r):
        return defer.Deferred()
    d.addCallback(neverFire)
    d.callback(1)
    return d