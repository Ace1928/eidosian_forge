import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbCompleted(ign):
    """
            Flush the exceptions which the reactor should have logged and make
            sure they're actually there.
            """
    errs = self.flushLoggedErrors(BadClientError)
    self.assertEqual(len(errs), 2, 'Incorrectly found %d errors, expected 2' % (len(errs),))