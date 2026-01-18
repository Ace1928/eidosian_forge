from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_identityPumpPolicy(self):
    """
        L{identityPumpPolicy} is a pump policy which calls the target's
        C{dataReceived} method one for each string in the queue passed to it.
        """
    bytes = []
    client = Protocol()
    client.dataReceived = bytes.append
    queue = loopback._LoopbackQueue()
    queue.put(b'foo')
    queue.put(b'bar')
    queue.put(None)
    loopback.identityPumpPolicy(queue, client)
    self.assertEqual(bytes, [b'foo', b'bar'])