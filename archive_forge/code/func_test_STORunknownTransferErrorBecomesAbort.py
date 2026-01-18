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
def test_STORunknownTransferErrorBecomesAbort(self):
    """
        Any non FTP error raised by STOR while transferring the file is
        converted into a critical error and transfer is closed.

        The unknown error is logged.
        """

    class FailingFileWriter(ftp._FileWriter):

        def receive(self):
            return defer.fail(AssertionError())

    def failingSTOR(a, b):
        return defer.succeed(FailingFileWriter(None))
    self.patch(ftp.FTPAnonymousShell, 'openForWriting', failingSTOR)

    def eb(res):
        res.trap(ftp.CommandFailed)
        logs = self.flushLoggedErrors()
        self.assertEqual(1, len(logs))
        self.assertIsInstance(logs[0].value, AssertionError)
        self.assertEqual(res.value.args[0][0], '426 Transfer aborted.  Data connection closed.')
    d1, d2 = self.client.storeFile('failing_file')
    d2.addErrback(eb)
    return defer.gatherResults([d1, d2])