import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_SHAHashA2AuthInt(self):
    """
        L{calcHA2} accepts the C{'sha'} algorithm and returns a SHA hash of
        its arguments, including the entity hash for QOP of C{'auth-int'}.
        """
    self.test_MD5HashA2AuthInt(b'sha', sha1)