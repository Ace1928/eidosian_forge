from __future__ import annotations
import socket
from twisted.internet import udp
from twisted.internet.protocol import DatagramProtocol
from twisted.python.runtime import platformType
from twisted.trial import unittest
def test_connectedReadImmediateError(self) -> None:
    """
        If the socket connected, socket reads with an immediate
        connection refusal are ignored, and reading stops. The protocol's
        C{connectionRefused} method is called.
        """
    udp._sockErrReadRefuse.append(-6000)
    self.addCleanup(udp._sockErrReadRefuse.remove, -6000)
    protocol = KeepReads()
    refused = []
    protocol.connectionRefused = lambda: refused.append(True)
    port = udp.Port(None, protocol)
    port.socket = StringUDPSocket([b'a', socket.error(-6000), b'b', socket.error(EWOULDBLOCK)])
    port.connect('127.0.0.1', 9999)
    port.doRead()
    self.assertEqual(protocol.reads, [b'a'])
    self.assertEqual(refused, [True])
    port.doRead()
    self.assertEqual(protocol.reads, [b'a', b'b'])
    self.assertEqual(refused, [True])