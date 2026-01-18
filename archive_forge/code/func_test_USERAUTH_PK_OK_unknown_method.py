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
def test_USERAUTH_PK_OK_unknown_method(self):
    """
        If C{SSHUserAuthClient} gets a MSG_USERAUTH_PK_OK packet when it's not
        expecting it, it should fail the current authentication and move on to
        the next type.
        """
    self.authClient.lastAuth = b'unknown'
    self.authClient.transport.packets = []
    self.authClient.ssh_USERAUTH_PK_OK(b'')
    self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])