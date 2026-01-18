import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def test_NoWarningOnBroadcast(self):
    """
        C{'<broadcast>'} is an alternative way to say C{'255.255.255.255'}
        ({socket.gethostbyname("<broadcast>")} returns C{'255.255.255.255'}),
        so because it becomes a valid IP address, no deprecation warning about
        passing hostnames to L{twisted.internet.udp.Port.write} needs to be
        emitted by C{write()} in this case.
        """

    class fakeSocket:

        def sendto(self, foo, bar):
            pass
    p = udp.Port(0, Server())
    p.socket = fakeSocket()
    p.write(b'test', ('<broadcast>', 1234))
    warnings = self.flushWarnings([self.test_NoWarningOnBroadcast])
    self.assertEqual(len(warnings), 0)