import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_factoryLogPrefix(self):
    """
        L{WrappingFactory.logPrefix} is customized to mention both the original
        factory and the wrapping factory.
        """
    server = Server()
    factory = policies.WrappingFactory(server)
    self.assertEqual('Server (WrappingFactory)', factory.logPrefix())