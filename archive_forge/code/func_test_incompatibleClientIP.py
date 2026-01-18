import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_incompatibleClientIP(self):
    """
        L{DigestCredentialFactory.decode} raises L{LoginFailed} when the
        request comes from a client IP other than what is encoded in the
        opaque.
        """
    credentialFactory = FakeDigestCredentialFactory(self.algorithm, self.realm)
    challenge = credentialFactory.getChallenge(self.clientAddress.host)
    badAddress = '10.0.0.1'
    self.assertNotEqual(self.clientAddress.host, badAddress)
    badNonceOpaque = credentialFactory._generateOpaque(challenge['nonce'], badAddress)
    self.assertRaises(LoginFailed, credentialFactory._verifyOpaque, badNonceOpaque, challenge['nonce'], self.clientAddress.host)