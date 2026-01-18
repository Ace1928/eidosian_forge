from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testConnectionMade():
    self.clientProtocol.transport.write(b'x' * nBytes)