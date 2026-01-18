from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_clientGreeting(self):
    """
        Test that on a connection where the client speaks first, the server
        receives the bytes sent by the client.
        """
    return self._greetingtest('write', False)