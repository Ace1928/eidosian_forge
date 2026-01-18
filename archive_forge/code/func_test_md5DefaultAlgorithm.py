import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_md5DefaultAlgorithm(self):
    """
        The algorithm defaults to MD5 if it is not supplied in the response.
        """
    self.algorithm = None
    self.test_response()