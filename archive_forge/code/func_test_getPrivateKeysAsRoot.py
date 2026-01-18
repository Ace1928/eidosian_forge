import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
def test_getPrivateKeysAsRoot(self) -> None:
    """
        L{OpenSSHFactory.getPrivateKeys} should switch to root if the keys
        aren't readable by the current user.
        """
    keyFile = self.keysDir.child('ssh_host_two_key')
    keyFile.chmod(0)
    self.addCleanup(keyFile.chmod, 511)
    savedSeteuid = os.seteuid

    def seteuid(euid: int) -> None:
        keyFile.chmod(511)
        return savedSeteuid(euid)
    self.patch(os, 'seteuid', seteuid)
    keys = self.factory.getPrivateKeys()
    self.assertEqual(len(keys), 2)
    keyTypes = keys.keys()
    self.assertEqual(set(keyTypes), {b'ssh-rsa', b'ssh-dss'})
    self.assertEqual(self.mockos.seteuidCalls, [0, os.geteuid()])
    self.assertEqual(self.mockos.setegidCalls, [0, os.getegid()])