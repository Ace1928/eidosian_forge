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
def test_getPublicKeyBadKeyError(self):
    """
        If L{keys.Key.fromFile} raises a L{keys.BadKeyError}, the
        L{SSHUserAuthClient.getPublicKey} tries again to get a public key by
        calling itself recursively.
        """
    options = ConchOptions()
    self.tmpdir.child('id_dsa.pub').setContent(keydata.publicDSA_openssh)
    dsaFile = self.tmpdir.child('id_dsa')
    dsaFile.setContent(keydata.privateDSA_openssh)
    options.identitys = [self.rsaFile.path, dsaFile.path]
    self.tmpdir.child('id_rsa.pub').setContent(b'not a key!')
    client = SSHUserAuthClient(b'user', options, None)
    key = client.getPublicKey()
    self.assertTrue(key.isPublic())
    self.assertEqual(key, Key.fromString(keydata.publicDSA_openssh))
    self.assertEqual(client.usedFiles, [self.rsaFile.path, dsaFile.path])