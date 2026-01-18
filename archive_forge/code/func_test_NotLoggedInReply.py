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
def test_NotLoggedInReply(self):
    """
        When not logged in, most commands other than USER and PASS should
        get NOT_LOGGED_IN errors, but some can be called before USER and PASS.
        """
    loginRequiredCommandList = ['CDUP', 'CWD', 'LIST', 'MODE', 'PASV', 'PWD', 'RETR', 'STRU', 'SYST', 'TYPE']
    loginNotRequiredCommandList = ['FEAT']

    def checkFailResponse(exception, command):
        failureResponseLines = exception.args[0]
        self.assertTrue(failureResponseLines[-1].startswith('530'), "%s - Response didn't start with 530: %r" % (command, failureResponseLines[-1]))

    def checkPassResponse(result, command):
        result = result[0]
        self.assertFalse(result.startswith('530'), '%s - Response start with 530: %r' % (command, result))
    deferreds = []
    for command in loginRequiredCommandList:
        deferred = self.client.queueStringCommand(command)
        self.assertFailure(deferred, ftp.CommandFailed)
        deferred.addCallback(checkFailResponse, command)
        deferreds.append(deferred)
    for command in loginNotRequiredCommandList:
        deferred = self.client.queueStringCommand(command)
        deferred.addCallback(checkPassResponse, command)
        deferreds.append(deferred)
    return defer.DeferredList(deferreds, fireOnOneErrback=True)