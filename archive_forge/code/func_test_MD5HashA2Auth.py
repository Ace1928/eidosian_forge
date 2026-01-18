import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_MD5HashA2Auth(self, _algorithm=b'md5', _hash=md5):
    """
        L{calcHA2} accepts the C{'md5'} algorithm and returns an MD5 hash of
        its arguments, excluding the entity hash for QOP other than
        C{'auth-int'}.
        """
    method = b'GET'
    hashA2 = calcHA2(_algorithm, method, self.uri, b'auth', None)
    a2 = method + b':' + self.uri
    expected = hexlify(_hash(a2).digest())
    self.assertEqual(hashA2, expected)