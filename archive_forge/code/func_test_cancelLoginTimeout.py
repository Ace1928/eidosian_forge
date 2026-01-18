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
def test_cancelLoginTimeout(self):
    """
        Test that stopping the service also stops the login timeout.
        """
    timeoutAuthServer = userauth.SSHUserAuthServer()
    timeoutAuthServer.clock = task.Clock()
    timeoutAuthServer.transport = FakeTransport(self.portal)
    timeoutAuthServer.serviceStarted()
    timeoutAuthServer.serviceStopped()
    timeoutAuthServer.clock.advance(11 * 60 * 60)
    self.assertEqual(timeoutAuthServer.transport.packets, [])
    self.assertFalse(timeoutAuthServer.transport.lostConnection)