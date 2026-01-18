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
def assertCommandResponse(self, command, expectedResponseLines, chainDeferred=None):
    """
        Asserts that a sending an FTP command receives the expected
        response.

        Returns a Deferred.  Optionally accepts a deferred to chain its actions
        to.
        """
    if chainDeferred is None:
        chainDeferred = defer.succeed(None)

    def queueCommand(ignored):
        d = self.client.queueStringCommand(command)

        def gotResponse(responseLines):
            self.assertEqual(expectedResponseLines, responseLines)
        return d.addCallback(gotResponse)
    return chainDeferred.addCallback(queueCommand)