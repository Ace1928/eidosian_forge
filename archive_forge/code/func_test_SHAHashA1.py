import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_SHAHashA1(self):
    """
        L{calcHA1} accepts the C{'sha'} algorithm and returns a SHA hash of its
        parameters, excluding the nonce and cnonce.
        """
    self.test_MD5HashA1(b'sha', sha1)