import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_checkHash(self):
    """
        L{DigestCredentialFactory.decode} returns an L{IUsernameDigestHash}
        provider which can verify a hash of the form 'username:realm:password'.
        """
    challenge = self.credentialFactory.getChallenge(self.clientAddress.host)
    nc = b'00000001'
    clientResponse = self.formatResponse(nonce=challenge['nonce'], response=self.getDigestResponse(challenge, nc), nc=nc, opaque=challenge['opaque'])
    creds = self.credentialFactory.decode(clientResponse, self.method, self.clientAddress.host)
    self.assertTrue(verifyObject(IUsernameDigestHash, creds))
    cleartext = self.username + b':' + self.realm + b':' + self.password
    hash = md5(cleartext)
    self.assertTrue(creds.checkHash(hexlify(hash.digest())))
    hash.update(b'wrong')
    self.assertFalse(creds.checkHash(hexlify(hash.digest())))