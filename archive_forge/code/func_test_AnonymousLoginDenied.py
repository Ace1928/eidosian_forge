import errno
import getpass
import os
import random
import string
from io import BytesIO
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm
from twisted.internet import defer, error, protocol, reactor, task
from twisted.internet.interfaces import IConsumer
from twisted.protocols import basic, ftp, loopback
from twisted.python import failure, filepath, runtime
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def test_AnonymousLoginDenied(self):
    """
        Reconfigure the server to disallow anonymous access, and to have an
        IUsernamePassword checker that always rejects.

        @return: L{Deferred} of command response
        """
    self.factory.allowAnonymous = False
    denyAlwaysChecker = checkers.InMemoryUsernamePasswordDatabaseDontUse()
    self.factory.portal.registerChecker(denyAlwaysChecker, credentials.IUsernamePassword)
    d = self.assertCommandResponse('USER anonymous', ['331 Password required for anonymous.'])
    d = self.assertCommandFailed('PASS test@twistedmatrix.com', ['530 Sorry, Authentication failed.'], chainDeferred=d)
    d = self.assertCommandFailed('PWD', ['530 Please login with USER and PASS.'], chainDeferred=d)
    return d