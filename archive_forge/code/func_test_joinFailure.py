import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
@skipIf(runtime.platform.isWindows() and (not runtime.platform.isVista()), "Windows' UDP multicast is not yet fully supported.")
def test_joinFailure(self):
    """
        Test that an attempt to join an address which is not a multicast
        address fails with L{error.MulticastJoinError}.
        """
    return self.assertFailure(self.client.transport.joinGroup('127.0.0.1'), error.MulticastJoinError)