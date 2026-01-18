from twisted.internet import address, defer, protocol, reactor
from twisted.protocols import portforward, wire
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
def testDataReceived(data):
    received.extend(iterbytes(data))
    if len(received) >= nBytes:
        self.assertEqual(b''.join(received), b'x' * nBytes)
        d.callback(None)