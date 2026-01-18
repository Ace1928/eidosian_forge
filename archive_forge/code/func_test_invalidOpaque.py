import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_invalidOpaque(self):
    """
        L{DigestCredentialFactory.decode} raises L{LoginFailed} when the opaque
        value does not contain all the required parts.
        """
    credentialFactory = FakeDigestCredentialFactory(self.algorithm, self.realm)
    challenge = credentialFactory.getChallenge(self.clientAddress.host)
    exc = self.assertRaises(LoginFailed, credentialFactory._verifyOpaque, b'badOpaque', challenge['nonce'], self.clientAddress.host)
    self.assertEqual(str(exc), 'Invalid response, invalid opaque value')
    badOpaque = b'foo-' + b64encode(b'nonce,clientip')
    exc = self.assertRaises(LoginFailed, credentialFactory._verifyOpaque, badOpaque, challenge['nonce'], self.clientAddress.host)
    self.assertEqual(str(exc), 'Invalid response, invalid opaque value')
    exc = self.assertRaises(LoginFailed, credentialFactory._verifyOpaque, b'', challenge['nonce'], self.clientAddress.host)
    self.assertEqual(str(exc), 'Invalid response, invalid opaque value')
    badOpaque = b'foo-' + b64encode(b','.join((challenge['nonce'], networkString(self.clientAddress.host), b'foobar')))
    exc = self.assertRaises(LoginFailed, credentialFactory._verifyOpaque, badOpaque, challenge['nonce'], self.clientAddress.host)
    self.assertEqual(str(exc), 'Invalid response, invalid opaque/time values')