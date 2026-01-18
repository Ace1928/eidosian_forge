import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
@skipIf(runtime.platform.isWindows(), 'on non-linux platforms it appears multiple processes can listen, but not multiple sockets in same process?')
def test_multiListen(self):
    """
        Test that multiple sockets can listen on the same multicast port and
        that they both receive multicast messages directed to that address.
        """
    firstClient = Server()
    firstPort = reactor.listenMulticast(0, firstClient, listenMultiple=True)
    portno = firstPort.getHost().port
    secondClient = Server()
    secondPort = reactor.listenMulticast(portno, secondClient, listenMultiple=True)
    theGroup = '225.0.0.250'
    joined = gatherResults([self.server.transport.joinGroup(theGroup), firstPort.joinGroup(theGroup), secondPort.joinGroup(theGroup)])

    def serverJoined(ignored):
        d1 = firstClient.packetReceived = Deferred()
        d2 = secondClient.packetReceived = Deferred()
        firstClient.transport.write(b'hello world', (theGroup, portno))
        return gatherResults([d1, d2])
    joined.addCallback(serverJoined)

    def gotPackets(ignored):
        self.assertEqual(firstClient.packets[0][0], b'hello world')
        self.assertEqual(secondClient.packets[0][0], b'hello world')
    joined.addCallback(gotPackets)

    def cleanup(passthrough):
        result = gatherResults([maybeDeferred(firstPort.stopListening), maybeDeferred(secondPort.stopListening)])
        result.addCallback(lambda ign: passthrough)
        return result
    joined.addBoth(cleanup)
    return joined