import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_multiResponse(self):
    """
        L{DigestCredentialFactory.decode} handles multiple responses to a
        single challenge.
        """
    challenge = self.credentialFactory.getChallenge(self.clientAddress.host)
    nc = b'00000001'
    clientResponse = self.formatResponse(nonce=challenge['nonce'], response=self.getDigestResponse(challenge, nc), nc=nc, opaque=challenge['opaque'])
    creds = self.credentialFactory.decode(clientResponse, self.method, self.clientAddress.host)
    self.assertTrue(creds.checkPassword(self.password))
    self.assertFalse(creds.checkPassword(self.password + b'wrong'))
    nc = b'00000002'
    clientResponse = self.formatResponse(nonce=challenge['nonce'], response=self.getDigestResponse(challenge, nc), nc=nc, opaque=challenge['opaque'])
    creds = self.credentialFactory.decode(clientResponse, self.method, self.clientAddress.host)
    self.assertTrue(creds.checkPassword(self.password))
    self.assertFalse(creds.checkPassword(self.password + b'wrong'))