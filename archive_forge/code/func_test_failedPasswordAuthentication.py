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
def test_failedPasswordAuthentication(self):
    """
        When provided with invalid authentication details, the server should
        respond by sending a MSG_USERAUTH_FAILURE message which states whether
        the authentication was partially successful, and provides other, open
        options for authentication.

        See RFC 4252, Section 5.1.
        """
    packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'bar')])
    self.authServer.clock = task.Clock()
    d = self.authServer.ssh_USERAUTH_REQUEST(packet)
    self.assertEqual(self.authServer.transport.packets, [])
    self.authServer.clock.advance(2)
    return d.addCallback(self._checkFailed)