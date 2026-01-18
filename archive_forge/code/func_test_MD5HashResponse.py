import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5HashResponse(self, _algorithm=b'md5', _hash=md5):
    """
        L{calcResponse} accepts the C{'md5'} algorithm and returns an MD5 hash
        of its parameters, excluding the nonce count, client nonce, and QoP
        value if the nonce count and client nonce are L{None}
        """
    hashA1 = b'abc123'
    hashA2 = b'789xyz'
    nonce = b'lmnopq'
    response = hashA1 + b':' + nonce + b':' + hashA2
    expected = hexlify(_hash(response).digest())
    digest = calcResponse(hashA1, hashA2, _algorithm, nonce, None, None, None)
    self.assertEqual(expected, digest)