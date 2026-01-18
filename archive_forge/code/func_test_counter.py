import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_counter(self):
    """
        Test counter management with the resetCounter method.
        """
    wrappedFactory = Server()
    f = TestLoggingFactory(wrappedFactory, 'test')
    self.assertEqual(f._counter, 0)
    f.buildProtocol(('1.2.3.4', 5678))
    self.assertEqual(f._counter, 1)
    f.openFile = None
    f.buildProtocol(('1.2.3.4', 5679))
    self.assertEqual(f._counter, 2)
    f.resetCounter()
    self.assertEqual(f._counter, 0)