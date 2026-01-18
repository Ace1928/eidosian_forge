import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_noNonce(self):
    """
        L{DigestCredentialFactory.decode} raises L{LoginFailed} if the response
        has no nonce.
        """
    e = self.assertRaises(LoginFailed, self.credentialFactory.decode, self.formatResponse(opaque=b'abc123'), self.method, self.clientAddress.host)
    self.assertEqual(str(e), 'Invalid response, no nonce given.')