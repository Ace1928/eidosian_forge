from twisted.application.internet import TCPClient
from twisted.trial import unittest
from twisted.words.protocols.jabber import jstrports
def test_client(self):
    """
        L{jstrports.client} returns a L{TCPClient} service.
        """
    got = jstrports.client('tcp:DOMAIN:65535', 'Factory')
    self.assertIsInstance(got, TCPClient)