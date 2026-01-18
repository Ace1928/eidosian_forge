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
def test_failedMakeDirectory(self):
    """
        L{ftp.FTPClient.makeDirectory} returns a L{Deferred} which is errbacked
        with L{CommandFailed} if the server returns an error response code.
        """
    self._testLogin()
    d = self.client.makeDirectory('/spam')
    self.assertEqual(self.transport.value(), b'MKD /spam\r\n')
    self.client.lineReceived(b'550 PERMISSION DENIED')
    return self.assertFailure(d, ftp.CommandFailed)