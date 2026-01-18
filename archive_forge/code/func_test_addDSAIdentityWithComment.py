import struct
from twisted.test import iosim
from twisted.trial import unittest
from twisted.conch.error import ConchError, MissingKeyStoreError
from twisted.conch.test import keydata
def test_addDSAIdentityWithComment(self):
    """
        L{SSHAgentClient.addIdentity} adds the private key it is called
        with to the SSH agent server to which it is connected, associating
        it with the comment it is called with.

        This test asserts that the server receives/stores the comment
        as sent by the client.
        """
    d = self.client.addIdentity(self.dsaPrivate.privateBlob(), comment=b'My special key')
    self.pump.flush()

    def _check(ignored):
        serverKey = self.server.factory.keys[self.dsaPrivate.blob()]
        self.assertEqual(self.dsaPrivate, serverKey[0])
        self.assertEqual(b'My special key', serverKey[1])
    return d.addCallback(_check)