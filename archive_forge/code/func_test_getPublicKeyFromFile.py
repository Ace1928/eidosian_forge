import sys
from unittest import skipIf
from twisted.conch.error import ConchError
from twisted.conch.test import keydata
from twisted.internet.testing import StringTransport
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_getPublicKeyFromFile(self):
    """
        L{SSHUserAuthClient.getPublicKey()} is able to get a public key from
        the first file described by its options' C{identitys} list, and return
        the corresponding public L{Key} object.
        """
    options = ConchOptions()
    options.identitys = [self.rsaFile.path]
    client = SSHUserAuthClient(b'user', options, None)
    key = client.getPublicKey()
    self.assertTrue(key.isPublic())
    self.assertEqual(key, self.rsaPublic)