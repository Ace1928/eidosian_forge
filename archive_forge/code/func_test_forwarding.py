from twisted.internet.testing import StringTransport
from twisted.protocols import finger
from twisted.trial import unittest
def test_forwarding(self) -> None:
    """
        When L{finger.Finger} receives a request for a remote user, it responds
        with a message rejecting the request.
        """
    self.protocol.dataReceived(b'moshez@example.com\r\n')
    self.assertEqual(self.transport.value(), b'Finger forwarding service denied\n')