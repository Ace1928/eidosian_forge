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
def test_downloadFolder(self):
    """
        When RETR is called for a folder, it will fail complaining that
        the path is a folder.
        """
    self.dirPath.child('foo').createDirectory()
    d = self._anonymousLogin()
    d.addCallback(self._makeDataConnection)

    def retrFolder(downloader):
        downloader.transport.loseConnection()
        deferred = self.client.queueStringCommand('RETR foo')
        return deferred
    d.addCallback(retrFolder)

    def failOnSuccess(result):
        raise AssertionError('Downloading a folder should not succeed.')
    d.addCallback(failOnSuccess)

    def checkError(failure):
        failure.trap(ftp.CommandFailed)
        self.assertEqual(['550 foo: is a directory'], failure.value.args[0])
        current_errors = self.flushLoggedErrors()
        self.assertEqual(0, len(current_errors), 'No errors should be logged while downloading a folder.')
    d.addErrback(checkError)
    return d