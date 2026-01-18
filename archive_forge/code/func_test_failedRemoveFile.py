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
def test_failedRemoveFile(self):
    """
        If the server returns a response code other than 250 in response to a
        I{DELE} sent by L{ftp.FTPClient.removeFile}, the L{Deferred} returned
        by C{removeFile} is errbacked with a L{Failure} wrapping a
        L{CommandFailed}.
        """
    self._testLogin()
    d = self.client.removeFile('/tmp/test')
    self.assertEqual(self.transport.value(), b'DELE /tmp/test\r\n')
    response = '501 Syntax error in parameters or arguments.'
    self.client.lineReceived(response.encode(self.client._encoding))
    d = self.assertFailure(d, ftp.CommandFailed)
    d.addCallback(lambda exc: self.assertEqual(exc.args, ([response],)))
    return d