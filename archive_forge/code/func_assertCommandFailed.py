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
def assertCommandFailed(self, command, expectedResponse=None, chainDeferred=None):
    if chainDeferred is None:
        chainDeferred = defer.succeed(None)

    def queueCommand(ignored):
        return self.client.queueStringCommand(command)
    chainDeferred.addCallback(queueCommand)
    self.assertFailure(chainDeferred, ftp.CommandFailed)

    def failed(exception):
        if expectedResponse is not None:
            self.assertEqual(expectedResponse, exception.args[0])
    return chainDeferred.addCallback(failed)