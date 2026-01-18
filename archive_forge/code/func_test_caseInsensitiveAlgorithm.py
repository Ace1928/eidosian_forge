import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_caseInsensitiveAlgorithm(self):
    """
        The case of the algorithm value in the response is ignored when
        checking the credentials.
        """
    self.algorithm = b'MD5'
    self.test_response()