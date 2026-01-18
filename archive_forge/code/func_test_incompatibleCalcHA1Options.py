import base64
from binascii import hexlify
from hashlib import md5, sha1
from zope.interface.verify import verifyObject
from twisted.cred.credentials import (
from twisted.cred.error import LoginFailed
from twisted.internet.address import IPv4Address
from twisted.python.compat import networkString
from twisted.trial.unittest import TestCase
def test_incompatibleCalcHA1Options(self):
    """
        L{calcHA1} raises L{TypeError} when any of the pszUsername, pszRealm,
        or pszPassword arguments are specified with the preHA1 keyword
        argument.
        """
    arguments = ((b'user', b'realm', b'password', b'preHA1'), (None, b'realm', None, b'preHA1'), (None, None, b'password', b'preHA1'))
    for pszUsername, pszRealm, pszPassword, preHA1 in arguments:
        self.assertRaises(TypeError, calcHA1, b'md5', pszUsername, pszRealm, pszPassword, b'nonce', b'cnonce', preHA1=preHA1)