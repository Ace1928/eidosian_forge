import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def getDigestResponse(self, challenge, ncount):
    """
        Calculate the response for the given challenge
        """
    nonce = challenge.get('nonce')
    algo = challenge.get('algorithm').lower()
    qop = challenge.get('qop')
    ha1 = calcHA1(algo, self.username, self.realm, self.password, nonce, self.cnonce)
    ha2 = calcHA2(algo, b'GET', self.uri, qop, None)
    expected = calcResponse(ha1, ha2, algo, nonce, ncount, self.cnonce, qop)
    return expected