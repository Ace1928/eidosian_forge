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
def test_getPrivateKeyPassphrase(self):
    """
        L{SSHUserAuthClient} can get a private key from a file, and return a
        Deferred called back with a private L{Key} object, even if the key is
        encrypted.
        """
    rsaPrivate = Key.fromString(keydata.privateRSA_openssh)
    passphrase = b'this is the passphrase'
    self.rsaFile.setContent(rsaPrivate.toString('openssh', passphrase=passphrase))
    options = ConchOptions()
    options.identitys = [self.rsaFile.path]
    client = SSHUserAuthClient(b'user', options, None)
    client.getPublicKey()

    def _getPassword(prompt):
        self.assertEqual(prompt, f"Enter passphrase for key '{self.rsaFile.path}': ")
        return nativeString(passphrase)

    def _cbGetPrivateKey(key):
        self.assertFalse(key.isPublic())
        self.assertEqual(key, rsaPrivate)
    self.patch(client, '_getPassword', _getPassword)
    return client.getPrivateKey().addCallback(_cbGetPrivateKey)