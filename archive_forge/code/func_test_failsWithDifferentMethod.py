import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_failsWithDifferentMethod(self):
    """
        L{DigestCredentialFactory.decode} returns an L{IUsernameHashedPassword}
        provider which rejects a correct password for the given user if the
        challenge response request is made using a different HTTP method than
        was used to request the initial challenge.
        """
    challenge = self.credentialFactory.getChallenge(self.clientAddress.host)
    nc = b'00000001'
    clientResponse = self.formatResponse(nonce=challenge['nonce'], response=self.getDigestResponse(challenge, nc), nc=nc, opaque=challenge['opaque'])
    creds = self.credentialFactory.decode(clientResponse, b'POST', self.clientAddress.host)
    self.assertFalse(creds.checkPassword(self.password))
    self.assertFalse(creds.checkPassword(self.password + b'wrong'))