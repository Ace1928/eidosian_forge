from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
def test_failedPrivateKeyAuthenticationWithoutSignature(self):
    """
        Test that private key authentication fails when the public key
        is invalid.
        """
    blob = keys.Key.fromString(keydata.publicDSA_openssh).blob()
    packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-dsa') + NS(blob)
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)
    return d.addCallback(self._checkFailed)