import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_SHAHashResponse(self):
    """
        L{calcResponse} accepts the C{'sha'} algorithm and returns a SHA hash
        of its parameters, excluding the nonce count, client nonce, and QoP
        value if the nonce count and client nonce are L{None}
        """
    self.test_MD5HashResponse(b'sha', sha1)