import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def test_loggingFactoryOpensLogfileAutomatically(self):
    """
        When the L{policies.TrafficLoggingFactory} builds a protocol, it
        automatically opens a unique log file for that protocol and attaches
        the logfile to the built protocol.
        """
    open_calls = []
    open_rvalues = []

    def mocked_open(*args, **kwargs):
        """
            Mock for the open call to prevent actually opening a log file.
            """
        open_calls.append((args, kwargs))
        io = StringIO()
        io.name = args[0]
        open_rvalues.append(io)
        return io
    self.patch(builtins, 'open', mocked_open)
    wrappedFactory = protocol.ServerFactory()
    wrappedFactory.protocol = SimpleProtocol
    factory = policies.TrafficLoggingFactory(wrappedFactory, 'test')
    first_proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 12345))
    second_proto = factory.buildProtocol(address.IPv4Address('TCP', '127.0.0.1', 12346))
    first_call = (('test-1', 'w'), {})
    second_call = (('test-2', 'w'), {})
    self.assertEqual([first_call, second_call], open_calls)
    self.assertEqual([first_proto.logfile, second_proto.logfile], open_rvalues)